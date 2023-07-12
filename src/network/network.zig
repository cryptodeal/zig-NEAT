const std = @import("std");
const net_node = @import("nnode.zig");
const net_math = @import("../math/activations.zig");
const common = @import("common.zig");
const network_graph = @import("../graph/graph.zig");

const NodeNeuronType = common.NodeNeuronType;
const Graph = network_graph.Graph;
const NNode = net_node.NNode;

pub const Network = struct {
    // network id]
    id: i64,
    // network name
    name: []const u8 = undefined,
    // NNodes that output from network
    outputs: []*NNode,

    // number of links in network (-1 means not yet counted)
    num_links: i64 = -1,

    // list of all NNodes in network (excluding MIMO control nodes)
    all_nodes: []*NNode,

    // NNodes that input into network
    inputs: []*NNode,

    // NNodes that connect network modules
    control_nodes: []*NNode = undefined,
    has_control_nodes: bool = false,

    // list of all nodes in the network including MIMO control ones
    all_nodes_MIMO: std.ArrayList(*NNode),

    // associated `Graph` struct
    graph: Graph(i64, void, f64) = undefined,

    // holds ref to allocator for use when freeing associated memory
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, in: []*NNode, out: []*NNode, all: []*NNode, net_id: i64) !*Network {
        var self: *Network = try allocator.create(Network);
        var graph = Graph(i64, void, f64).init(allocator);
        for (all) |node| {
            for (node.outgoing.items) |edge| {
                try graph.add_edge(edge.in_node.?.id, {}, edge.out_node.?.id, {}, edge.cxn_weight);
            }
        }
        self.* = .{
            .allocator = allocator,
            .id = net_id,
            .inputs = in,
            .outputs = out,
            .all_nodes = all,
            .graph = graph,
            .num_links = -1,
            .all_nodes_MIMO = std.ArrayList(*NNode).init(allocator),
        };

        try self.all_nodes_MIMO.appendSlice(all);
        return self;
    }

    pub fn init_modular(allocator: std.mem.Allocator, in: []*NNode, out: []*NNode, all: []*NNode, control: []*NNode, net_id: i64) !*Network {
        var self: *Network = try Network.init(allocator, in, out, all, net_id);
        self.control_nodes = control;
        self.has_control_nodes = true;
        for (control) |node| {
            for (node.outgoing.items) |edge| {
                try self.graph.add_edge(edge.in_node.?.id, {}, edge.out_node.?.id, {}, edge.cxn_weight);
            }
        }
        try self.all_nodes_MIMO.appendSlice(control);
        return self;
    }

    pub fn deinit(self: *Network) void {
        if (self.has_control_nodes) {
            for (self.control_nodes) |n| {
                for (n.outgoing.items) |l| {
                    l.deinit();
                }
            }
            self.allocator.free(self.control_nodes);
        }
        for (self.all_nodes_MIMO.items) |n| {
            for (n.incoming.items) |l| {
                l.deinit();
            }
            n.deinit();
        }
        self.allocator.free(self.all_nodes);
        self.allocator.free(self.inputs);
        self.allocator.free(self.outputs);
        self.all_nodes_MIMO.deinit();
        self.graph.deinit();
        self.allocator.destroy(self);
    }

    pub fn node_id_generator(self: *Network) i64 {
        return @as(i64, @intCast(self.all_nodes.len));
    }

    fn process_list(_: *Network, start_idx: usize, n_list: []*NNode, activations: []net_math.NodeActivationType, neuron_lookup: *std.AutoHashMap(i64, i64)) !usize {
        var idx = start_idx;
        for (n_list) |ne| {
            activations[start_idx] = ne.activation_type;
            try neuron_lookup.put(ne.id, @as(i64, @intCast(start_idx)));
            idx += 1;
        }
        return idx;
    }

    pub fn is_control_node(self: *Network, nid: i64) bool {
        for (self.control_nodes) |cn| {
            if (cn.id == nid) {
                return true;
            }
        }
        return false;
    }

    pub fn flush(self: *Network) !bool {
        for (self.all_nodes) |node| {
            node.flushback();
            try node.flushback_check();
        }
        return true;
    }

    pub fn print_activation(self: *Network, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().print("Network {s} with id {d} outputs: (", .{ self.name, self.id });
        for (self.outputs, 0..) |node, i| {
            try buffer.writer().print("[Output #{d}: {any}] ", .{ i, node });
        }
        try buffer.writer().print(")", .{});
        return buffer.toOwnedSlice();
    }

    pub fn print_input(self: *Network, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().print("Network {s} with id {d} inputs: (", .{ self.name, self.id });
        for (self.inputs, 0..) |node, i| {
            try buffer.writer().print("[Input #{d}: {any}] ", .{ i, node });
        }
        try buffer.writer().print(")", .{});
        return buffer.toOwnedSlice();
    }

    pub fn output_is_off(self: *Network) bool {
        for (self.outputs) |node| {
            if (node.activations_count == 0) {
                return true;
            }
        }
        return false;
    }

    pub fn activate_steps(self: *Network, max_steps: i64) !bool {
        if (max_steps == 0) {
            return error.ErrZeroActivationStepsRequested;
        }

        // for adding active sum
        var add_amount: f64 = 0.0;
        // ensure activate at least once
        var one_time = false;
        // used if output is truncated from network
        var abort_count: i64 = 0;

        // loop until all the outputs are activated
        while (self.output_is_off() or !one_time) {
            if (abort_count >= max_steps) {
                return error.ErrNetExceededMaxActivationAttempts;
            }

            for (self.all_nodes) |np| {
                if (np.is_neuron()) {
                    // reset activation value
                    np.activation_sum = 0.0;

                    // For each node's incoming connection, add the activity from the connection to the activesum
                    for (np.incoming.items) |link| {
                        // handle potential time delayed cxns
                        if (!link.is_time_delayed) {
                            add_amount = link.cxn_weight * link.in_node.?.get_active_out();
                            if (link.in_node.?.is_active or link.in_node.?.is_sensor()) {
                                np.is_active = true;
                            }
                        } else {
                            add_amount = link.cxn_weight * link.in_node.?.get_active_out_td();
                        }
                        np.activation_sum += add_amount;
                    }
                }
            }

            // activate all the neuron nodes off their incoming activation
            // only activate if some active input recvd
            for (self.all_nodes) |np| {
                if (np.is_neuron() and np.is_active) {
                    try common.activate_node(np);
                }
            }

            // activate all MIMO control genes to propagate activation through genome modules
            if (self.has_control_nodes) {
                for (self.control_nodes) |cn| {
                    cn.is_active = false;
                    // activate control MIMO node as control module
                    try common.activate_module(cn);
                    cn.is_active = true;
                }
            }

            one_time = true;
            abort_count += 1;
        }

        return true;
    }

    pub fn activate(self: *Network) !bool {
        return self.activate_steps(20);
    }

    pub fn forward_steps(self: *Network, steps: i64) !bool {
        if (steps == 0) {
            return error.ErrZeroActivationStepsRequested;
        }
        var i: usize = 0;
        while (i < steps) : (i += 1) {
            _ = try self.activate_steps(steps);
        }
        return true;
    }

    pub fn recursive_steps(self: *Network) !bool {
        var net_depth = try self.max_activation_depth_fast(0);
        return self.forward_steps(net_depth);
    }

    pub fn relax() !bool {
        std.debug.print("relax not implemented\n", .{});
        return error.ErrNotImplemented;
    }

    pub fn load_sensors(self: *Network, sensors: []f64) void {
        var counter: usize = 0;
        if (sensors.len == self.inputs.len) {
            // BIAS value provided as input
            for (self.inputs) |node| {
                if (node.is_sensor()) {
                    _ = node.sensor_load(sensors[counter]);
                    counter += 1;
                }
            }
        } else {
            // use default BIAS value
            for (self.inputs) |node| {
                if (node.neuron_type == NodeNeuronType.InputNeuron) {
                    _ = node.sensor_load(sensors[counter]);
                    counter += 1;
                } else {
                    // default BIAS value
                    _ = node.sensor_load(1.0);
                }
            }
        }
    }

    pub fn read_outputs(self: *Network, allocator: std.mem.Allocator) ![]f64 {
        var outs = try allocator.alloc(f64, self.outputs.len);
        for (self.outputs, 0..) |o, i| {
            outs[i] = o.activation;
        }
        return outs;
    }

    pub fn node_count(self: *Network) i64 {
        if (!self.has_control_nodes or self.control_nodes.len == 0) {
            return @as(i64, @intCast(self.all_nodes.len));
        } else {
            return @as(i64, @intCast(self.all_nodes.len + self.control_nodes.len));
        }
    }

    pub fn link_count(self: *Network) i64 {
        self.num_links = 0;
        for (self.all_nodes) |node| {
            self.num_links += @as(i64, @intCast(node.incoming.items.len));
        }
        if (self.has_control_nodes and self.control_nodes.len != 0) {
            for (self.control_nodes) |node| {
                self.num_links += @as(i64, @intCast(node.incoming.items.len));
                self.num_links += @as(i64, @intCast(node.outgoing.items.len));
            }
        }
        return self.num_links;
    }

    pub fn complexity(self: *Network) i64 {
        return self.node_count() + self.link_count();
    }

    pub fn is_recurrent(self: *Network, in_node: *NNode, out_node: *NNode, count: *i64, thresh: i64) bool {
        // count node as visited
        count.* += 1;
        if (count.* > thresh) {
            // short out whole thing - loop detected
            return false;
        }
        if (std.meta.eql(in_node, out_node)) {
            return true;
        } else {
            // Check back on all links ...
            for (in_node.incoming.items) |link| {
                // skip links that are marked recurrent
                // only want to check back through the forward flow of signals
                if (!link.is_recurrent) {
                    if (self.is_recurrent(link.in_node.?, out_node, count, thresh)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    pub fn max_activation_depth(self: *Network) !i64 {
        if (self.all_nodes.len == self.inputs.len + self.outputs.len and !self.has_control_nodes or self.control_nodes.len == 0) {
            return 1;
        }

        return self.calc_max_activation_depth();
    }

    pub fn max_activation_depth_fast(self: *Network, max_depth: i64) !i64 {
        if (self.has_control_nodes and self.control_nodes.len > 0) {
            std.debug.print("unsupported for modular networks", .{});
            return error.ErrModularNetworkUnsupported;
        }

        if (self.all_nodes.len == self.inputs.len + self.outputs.len and !self.has_control_nodes or self.control_nodes.len == 0) {
            return 1;
        }

        var max: i64 = 0;
        for (self.outputs) |node| {
            var curr_depth = try node.depth(1, max_depth);
            if (curr_depth > max) {
                max = curr_depth;
            }
        }

        return max;
    }

    pub fn get_all_nodes(self: *Network) []*NNode {
        return self.all_nodes_MIMO.items;
    }

    pub fn get_control_nodes(self: *Network) []*NNode {
        return self.control_nodes;
    }

    pub fn calc_max_activation_depth(self: *Network) !i64 {
        var all_paths = self.graph.johnson_all_paths() catch try self.graph.floyd_warshall();
        var max: usize = 0;
        for (self.inputs) |in| {
            for (self.outputs) |out| {
                var paths = try all_paths.all_between(in.id, out.id);
                if (paths.paths != null) {
                    for (paths.paths.?.items) |p| {
                        var l = p.items.len;
                        if (l > max) {
                            max = l;
                        }
                    }
                }
            }
        }
        return @as(i64, @intCast(max));
    }
};
