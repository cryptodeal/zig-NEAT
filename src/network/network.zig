const std = @import("std");
const net_node = @import("nnode.zig");
const net_math = @import("../math/activations.zig");
const common = @import("common.zig");
const network_graph = @import("../graph/graph.zig");
const fast_net = @import("fast_network.zig");

const Solver = @import("solver.zig").Solver;
const NodeNeuronType = common.NodeNeuronType;
const Graph = network_graph.Graph;
const NNode = net_node.NNode;
const FastNetworkLink = fast_net.FastNetworkLink;
const FastControlNode = fast_net.FastControlNode;
const FastModularNetworkSolver = fast_net.FastModularNetworkSolver;

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
            for (node.incoming.items) |edge| {
                try self.graph.add_edge(edge.in_node.?.id, {}, edge.out_node.?.id, {}, edge.cxn_weight);
            }
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

    pub fn fast_network_solver(self: *Network, allocator: std.mem.Allocator) !Solver {
        // calculate neurons per layer
        const output_neuron_count = self.outputs.len;

        // build bias, input and hidden neurons lists
        var bias_neuron_count: usize = 0;
        var in_list = std.ArrayList(*NNode).init(allocator);
        defer in_list.deinit();
        var bias_list = std.ArrayList(*NNode).init(allocator);
        defer bias_list.deinit();
        var hidden_list = std.ArrayList(*NNode).init(allocator);
        defer hidden_list.deinit();
        for (self.all_nodes) |ne| {
            switch (ne.neuron_type) {
                .BiasNeuron => {
                    bias_neuron_count += 1;
                    try bias_list.append(ne);
                },
                .InputNeuron => try in_list.append(ne),
                .HiddenNeuron => try hidden_list.append(ne),
                else => continue,
            }
        }

        const input_neuron_count = in_list.items.len;
        const total_neuron_count = self.all_nodes.len;

        // create activation functions array
        var activations = try allocator.alloc(net_math.NodeActivationType, total_neuron_count);
        var neuron_lookup = std.AutoHashMap(i64, usize).init(allocator); // id:index
        defer neuron_lookup.deinit();

        // walk through neuron nodes in order: bias, input, output, hidden
        var neuron_idx = try self.process_list(0, bias_list.items, activations, &neuron_lookup);
        neuron_idx = try self.process_list(neuron_idx, in_list.items, activations, &neuron_lookup);
        neuron_idx = try self.process_list(neuron_idx, self.outputs, activations, &neuron_lookup);
        _ = try self.process_list(neuron_idx, hidden_list.items, activations, &neuron_lookup);

        // walk through neurons in order: input, output, hidden and create bias and connections lists
        var biases = try allocator.alloc(f64, total_neuron_count);
        for (biases) |*v| v.* = 0;

        var connections = std.ArrayList(*FastNetworkLink).init(allocator);

        var in_connects = try self.process_incoming_connections(allocator, in_list.items, biases, &neuron_lookup);
        try connections.appendSlice(in_connects);
        allocator.free(in_connects);
        in_connects = try self.process_incoming_connections(allocator, hidden_list.items, biases, &neuron_lookup);
        try connections.appendSlice(in_connects);
        allocator.free(in_connects);
        in_connects = try self.process_incoming_connections(allocator, self.outputs, biases, &neuron_lookup);
        try connections.appendSlice(in_connects);
        allocator.free(in_connects);

        // walk through control neurons
        var mod_count = if (self.has_control_nodes) self.control_nodes.len else 0;
        var modules = try allocator.alloc(*FastControlNode, mod_count);
        if (self.has_control_nodes) {
            for (self.control_nodes, 0..) |cn, i| {
                // collect inputs
                var inputs = try allocator.alloc(usize, cn.incoming.items.len);
                for (cn.incoming.items, 0..) |in, j| {
                    var in_i = neuron_lookup.get(in.in_node.?.id);
                    if (in_i != null) {
                        var in_idx = in_i.?;
                        inputs[j] = in_idx;
                    } else {
                        std.debug.print("failed lookup for input neuron with id {d} at control neuron {d}\n", .{ in.in_node.?.id, cn.id });
                        return error.NeuronLookupFailed;
                    }
                }
                // collect outputs
                var outputs = try allocator.alloc(usize, cn.outgoing.items.len);
                for (cn.outgoing.items, 0..) |out, j| {
                    var out_i = neuron_lookup.get(out.out_node.?.id);
                    if (out_i != null) {
                        var out_idx = out_i.?;
                        outputs[j] = out_idx;
                    } else {
                        std.debug.print("failed lookup for output neuron with id {d} at control neuron {d}\n", .{ out.out_node.?.id, cn.id });
                        return error.NeuronLookupFailed;
                    }
                }
                // build control node
                modules[i] = try FastControlNode.init(allocator, inputs, outputs, cn.activation_type);
            }
        }
        var modular_solver = try FastModularNetworkSolver.init(allocator, bias_neuron_count, input_neuron_count, output_neuron_count, total_neuron_count, activations, try connections.toOwnedSlice(), biases, modules);
        return Solver.init(modular_solver);
    }

    pub fn node_id_generator(self: *Network) i64 {
        return @as(i64, @intCast(self.all_nodes.len));
    }

    fn process_list(_: *Network, start_idx: usize, n_list: []*NNode, activations: []net_math.NodeActivationType, neuron_lookup: *std.AutoHashMap(i64, usize)) !usize {
        var idx = start_idx;
        for (n_list) |ne| {
            activations[idx] = ne.activation_type;
            try neuron_lookup.put(ne.id, idx);
            idx += 1;
        }
        return idx;
    }

    pub fn process_incoming_connections(_: *Network, allocator: std.mem.Allocator, n_list: []*NNode, biases: []f64, neuron_lookup: *std.AutoHashMap(i64, usize)) ![]*FastNetworkLink {
        var connections = std.ArrayList(*FastNetworkLink).init(allocator);
        for (n_list) |ne| {
            var target_i = neuron_lookup.get(ne.id);
            if (target_i != null) {
                var target_idx = target_i.?;
                for (ne.incoming.items) |in| {
                    var src_i = neuron_lookup.get(in.in_node.?.id);
                    if (src_i != null) {
                        var src_idx = src_i.?;
                        if (in.in_node.?.neuron_type == .BiasNeuron) {
                            // store bias for target neuron
                            biases[target_idx] += in.cxn_weight;
                        } else {
                            // save connection
                            var conn = try FastNetworkLink.init(allocator, src_idx, target_idx, in.cxn_weight);
                            try connections.append(conn);
                        }
                    } else {
                        std.debug.print("failed lookup for source neuron with id {d}\n", .{in.in_node.?.id});
                        return error.NeuronLookupFailed;
                    }
                }
            } else {
                std.debug.print("failed lookup for target neuron with id {d}\n", .{ne.id});
                return error.NeuronLookupFailed;
            }
        }
        return connections.toOwnedSlice();
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
                    } // End {for} over incoming links
                } // End if != SENSOR
            } // End {for} over all nodes

            // activate all the neuron nodes off their incoming activation
            // only activate if some active input recvd
            for (self.all_nodes) |np| {
                if (np.is_neuron()) {
                    // Only activate if some active input came in
                    if (np.is_active) {
                        try common.activate_node(np);
                    }
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
        var net_depth = try self.max_activation_depth_capped(0);
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
        if (!self.has_control_nodes or self.control_nodes.len == 0) {
            return self.max_activation_depth_capped(0);
        } else {
            return self.calc_max_activation_depth();
        }
    }

    pub fn max_activation_depth_capped(self: *Network, max_depth_cap: i64) !i64 {
        if (self.has_control_nodes and self.control_nodes.len > 0) {
            std.debug.print("unsupported for modular networks", .{});
            return error.ErrModularNetworkUnsupported;
        }

        if (self.all_nodes.len == self.inputs.len + self.outputs.len and !self.has_control_nodes or self.control_nodes.len == 0) {
            return 1;
        }

        var max: i64 = 0; // the max depth
        for (self.outputs) |node| {
            var curr_depth = node.depth(0, max_depth_cap) catch return max_depth_cap;
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

    pub fn get_base_nodes(self: *Network) []*NNode {
        return self.all_nodes;
    }

    pub fn calc_max_activation_depth(self: *Network) !i64 {
        var all_paths = self.graph.johnson_all_paths() catch try self.graph.floyd_warshall();
        defer all_paths.deinit();
        var max: usize = 0;
        for (self.inputs) |in| {
            for (self.outputs) |out| {
                var paths = try all_paths.all_between(in.id, out.id);
                defer paths.deinit();
                if (paths.paths != null) {
                    // iterate over returned paths and find the one with maximal length
                    for (paths.paths.?.items) |p| {
                        var l = p.items.len - 1; // to exclude input node
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

// test utils
fn build_plain_network(allocator: std.mem.Allocator) !*Network {
    var all_nodes = try allocator.alloc(*NNode, 5);
    all_nodes[0] = try NNode.init(allocator, 1, .InputNeuron);
    all_nodes[1] = try NNode.init(allocator, 2, .InputNeuron);
    all_nodes[2] = try NNode.init(allocator, 3, .BiasNeuron);
    all_nodes[3] = try NNode.init(allocator, 7, .OutputNeuron);
    all_nodes[4] = try NNode.init(allocator, 8, .OutputNeuron);

    // OUTPUT 7
    _ = try all_nodes[3].connect_from(allocator, all_nodes[1], 7);
    _ = try all_nodes[3].connect_from(allocator, all_nodes[2], 4.5);
    // OUTPUT 8
    _ = try all_nodes[4].connect_from(allocator, all_nodes[3], 13);

    var in = try allocator.alloc(*NNode, 3);
    @memcpy(in, all_nodes[0..3]);
    var out = try allocator.alloc(*NNode, 2);
    @memcpy(out, all_nodes[3..5]);

    return Network.init(allocator, in, out, all_nodes, 0);
}

fn build_disconnected_network(allocator: std.mem.Allocator) !*Network {
    var all_nodes = try allocator.alloc(*NNode, 8);
    all_nodes[0] = try NNode.init(allocator, 1, .InputNeuron);
    all_nodes[1] = try NNode.init(allocator, 2, .InputNeuron);
    all_nodes[2] = try NNode.init(allocator, 3, .BiasNeuron);
    all_nodes[3] = try NNode.init(allocator, 4, .HiddenNeuron);
    all_nodes[4] = try NNode.init(allocator, 5, .HiddenNeuron);
    all_nodes[5] = try NNode.init(allocator, 6, .HiddenNeuron);
    all_nodes[6] = try NNode.init(allocator, 7, .OutputNeuron);
    all_nodes[7] = try NNode.init(allocator, 8, .OutputNeuron);

    var in = try allocator.alloc(*NNode, 3);
    @memcpy(in, all_nodes[0..3]);
    var out = try allocator.alloc(*NNode, 2);
    @memcpy(out, all_nodes[6..8]);

    return Network.init(allocator, in, out, all_nodes, 0);
}

pub fn build_network(allocator: std.mem.Allocator) !*Network {
    var all_nodes = try allocator.alloc(*NNode, 8);
    all_nodes[0] = try NNode.init(allocator, 1, .InputNeuron);
    all_nodes[1] = try NNode.init(allocator, 2, .InputNeuron);
    all_nodes[2] = try NNode.init(allocator, 3, .BiasNeuron);
    all_nodes[3] = try NNode.init(allocator, 4, .HiddenNeuron);
    all_nodes[4] = try NNode.init(allocator, 5, .HiddenNeuron);
    all_nodes[5] = try NNode.init(allocator, 6, .HiddenNeuron);
    all_nodes[6] = try NNode.init(allocator, 7, .OutputNeuron);
    all_nodes[7] = try NNode.init(allocator, 8, .OutputNeuron);

    // HIDDEN 4
    _ = try all_nodes[3].connect_from(allocator, all_nodes[0], 15);
    _ = try all_nodes[3].connect_from(allocator, all_nodes[1], 10);
    // HIDDEN 5
    _ = try all_nodes[4].connect_from(allocator, all_nodes[1], 5);
    _ = try all_nodes[4].connect_from(allocator, all_nodes[2], 1);
    // HIDDEN 6
    _ = try all_nodes[5].connect_from(allocator, all_nodes[4], 17);
    // OUTPUT 7
    _ = try all_nodes[6].connect_from(allocator, all_nodes[3], 7);
    _ = try all_nodes[6].connect_from(allocator, all_nodes[5], 4.5);
    // OUTPUT 8
    _ = try all_nodes[7].connect_from(allocator, all_nodes[5], 13);

    var in = try allocator.alloc(*NNode, 3);
    @memcpy(in, all_nodes[0..3]);
    var out = try allocator.alloc(*NNode, 2);
    @memcpy(out, all_nodes[6..8]);

    return Network.init(allocator, in, out, all_nodes, 0);
}

pub fn build_modular_network(allocator: std.mem.Allocator) !*Network {
    var all_nodes = try allocator.alloc(*NNode, 8);
    all_nodes[0] = try NNode.init(allocator, 1, .InputNeuron); // INPUT 1
    all_nodes[1] = try NNode.init(allocator, 2, .InputNeuron); // INPUT 2
    all_nodes[2] = try NNode.init(allocator, 3, .BiasNeuron); // BIAS
    all_nodes[3] = try NNode.init(allocator, 4, .HiddenNeuron); // HIDDEN 4
    all_nodes[4] = try NNode.init(allocator, 5, .HiddenNeuron); // HIDDEN 5
    all_nodes[5] = try NNode.init(allocator, 7, .HiddenNeuron); // HIDDEN 7
    all_nodes[6] = try NNode.init(allocator, 8, .OutputNeuron);
    all_nodes[7] = try NNode.init(allocator, 9, .OutputNeuron);

    var control_nodes = try allocator.alloc(*NNode, 1);
    control_nodes[0] = try NNode.init(allocator, 6, .HiddenNeuron);
    // HIDDEN 6 - control node
    control_nodes[0].activation_type = .MultiplyModuleActivation;
    _ = try control_nodes[0].add_incoming(allocator, all_nodes[3], 1); // <- HIDDEN 4
    _ = try control_nodes[0].add_incoming(allocator, all_nodes[4], 1); // <- HIDDEN 5
    _ = try control_nodes[0].add_outgoing(allocator, all_nodes[5], 1); // <- HIDDEN 5

    // HIDDEN 4
    all_nodes[3].activation_type = .LinearActivation;
    _ = try all_nodes[3].connect_from(allocator, all_nodes[0], 15); // <- INPUT 1
    _ = try all_nodes[3].connect_from(allocator, all_nodes[2], 10); // <- BIAS
    // HIDDEN 5
    all_nodes[4].activation_type = .LinearActivation;
    _ = try all_nodes[4].connect_from(allocator, all_nodes[1], 5); // <- INPUT 2
    _ = try all_nodes[4].connect_from(allocator, all_nodes[2], 1); // <- BIAS

    // HIDDEN 7
    all_nodes[5].activation_type = .NullActivation;

    // OUTPUT 8
    all_nodes[6].activation_type = .LinearActivation;
    _ = try all_nodes[6].connect_from(allocator, all_nodes[5], 4.5); // <- HIDDEN 7
    // OUTPUT 9
    all_nodes[7].activation_type = .LinearActivation;
    _ = try all_nodes[7].connect_from(allocator, all_nodes[5], 13); // <- HIDDEN 7

    var in = try allocator.alloc(*NNode, 3);
    @memcpy(in, all_nodes[0..3]);
    var out = try allocator.alloc(*NNode, 2);
    @memcpy(out, all_nodes[6..8]);
    return Network.init_modular(allocator, in, out, all_nodes, control_nodes, 0);
}

test "Modular Network activate" {
    var net = try build_modular_network(std.testing.allocator);
    defer net.deinit();
    var data = [_]f64{ 1, 2, 1 };
    net.load_sensors(&data);
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        var res = try net.activate();
        try std.testing.expect(res);
    }

    try std.testing.expect(net.outputs[0].activation == 1237.5);
    try std.testing.expect(net.outputs[1].activation == 3575.0);
}

test "Network max activation depth (simple)" {
    var net = try build_network(std.testing.allocator);
    defer net.deinit();
    var depth = try net.max_activation_depth();
    try std.testing.expect(depth == 3);

    // TODO: log network activation path
}

test "Modular Network max activation depth" {
    var net = try build_modular_network(std.testing.allocator);
    defer net.deinit();
    var depth = try net.max_activation_depth();
    try std.testing.expect(depth == 4);

    // TODO: log network activation path
}

test "Network max activation depth (no hidden or control)" {
    var net = try build_plain_network(std.testing.allocator);
    defer net.deinit();
    var depth = try net.max_activation_depth();
    try std.testing.expect(depth == 1);
}

test "Network max activation depth capped (simple)" {
    var net = try build_network(std.testing.allocator);
    defer net.deinit();
    var depth = try net.max_activation_depth_capped(0);
    try std.testing.expect(depth == 3);

    // TODO: log network activation path

}

test "Network max activation depth (simple - exceed cap)" {
    var net = try build_network(std.testing.allocator);
    defer net.deinit();
    var depth = try net.max_activation_depth_capped(2);
    try std.testing.expect(depth == 2);
}

test "Modular Network max activation depth capped" {
    var net = try build_modular_network(std.testing.allocator);
    defer net.deinit();
    try std.testing.expectError(error.ErrModularNetworkUnsupported, net.max_activation_depth_capped(0));
}

test "Network output is off" {
    var net = try build_network(std.testing.allocator);
    defer net.deinit();
    try std.testing.expect(net.output_is_off());
}

test "Network activate" {
    var net = try build_network(std.testing.allocator);
    defer net.deinit();

    var res = try net.activate();
    try std.testing.expect(res);

    // check activation
    for (net.all_nodes) |node| {
        if (node.is_neuron()) {
            try std.testing.expect(node.activations_count != 0);
            try std.testing.expect(node.activation != 0);

            // Check activation and time delayed activation
            try std.testing.expect(node.get_active_out() != 0);
        }
    }
}

test "Network forward steps" {
    const allocator = std.testing.allocator;
    var net = try build_network(allocator);
    defer net.deinit();

    var res = try net.forward_steps(3);
    try std.testing.expect(res);

    var expected_outs = [_]f64{ 1, 1 };
    var net_outs = try net.read_outputs(allocator);
    defer allocator.free(net_outs);
    try std.testing.expect(net_outs.len == expected_outs.len);
    try std.testing.expectEqualSlices(f64, &expected_outs, net_outs);

    // test zero steps
    try std.testing.expectError(error.ErrZeroActivationStepsRequested, net.forward_steps(0));
}

test "Network recursive steps" {
    const allocator = std.testing.allocator;
    var net = try build_network(allocator);
    defer net.deinit();

    var data = [_]f64{ 0.5, 0, 1.5 };
    net.load_sensors(&data);

    var relaxed = try net.recursive_steps();
    try std.testing.expect(relaxed);

    // TODO: log network activation path
}

test "Network forward steps (disconnected)" {
    const allocator = std.testing.allocator;
    var net = try build_disconnected_network(allocator);
    defer net.deinit();
    try std.testing.expectError(error.ErrNetExceededMaxActivationAttempts, net.forward_steps(3));
}

test "Network load sensors" {
    const allocator = std.testing.allocator;
    var net = try build_network(allocator);
    defer net.deinit();

    var sensors = [_]f64{ 1, 3.4, 5.6 };
    net.load_sensors(&sensors);
    var counter: usize = 0;
    for (net.get_all_nodes()) |node| {
        if (node.is_sensor()) {
            try std.testing.expect(sensors[counter] == node.activation);
            try std.testing.expect(node.activations_count == 1);
            counter += 1;
        }
    }
}

test "Network flush" {
    const allocator = std.testing.allocator;
    var net = try build_network(allocator);
    defer net.deinit();

    // activate and check state
    var res = try net.activate();
    try std.testing.expect(res);

    // flush and check
    res = try net.flush();
    try std.testing.expect(res);

    for (net.get_all_nodes()) |node| {
        try std.testing.expect(node.activations_count == 0);
        try std.testing.expect(node.activation == 0);

        // Check activation and time delayed activation
        try std.testing.expect(node.get_active_out() == 0);
        try std.testing.expect(node.get_active_out_td() == 0);
    }
}

test "Network node count" {
    const allocator = std.testing.allocator;
    var net = try build_network(allocator);
    defer net.deinit();

    try std.testing.expect(net.node_count() == 8);
}

test "Network link count" {
    const allocator = std.testing.allocator;
    var net = try build_network(allocator);
    defer net.deinit();

    try std.testing.expect(net.link_count() == 8);
}

test "Network fast network solver" {
    const allocator = std.testing.allocator;
    var net = try build_network(allocator);
    defer net.deinit();

    var solver = try net.fast_network_solver(allocator);
    defer solver.deinit();

    // check solver structure
    try std.testing.expect(solver.node_count() == @as(usize, @intCast(net.node_count())));
    try std.testing.expect(solver.link_count() == @as(usize, @intCast(net.link_count())));
}

test "Network activate steps with zero activation steps" {
    const allocator = std.testing.allocator;
    var net = try build_network(allocator);
    defer net.deinit();
    try std.testing.expectError(error.ErrZeroActivationStepsRequested, net.activate_steps(0));
}

test "Network activate steps exceed max activation attempts" {
    const allocator = std.testing.allocator;
    var net = try build_disconnected_network(allocator);
    defer net.deinit();
    try std.testing.expectError(error.ErrNetExceededMaxActivationAttempts, net.activate_steps(3));
}

test "Modular Network control nodes" {
    const allocator = std.testing.allocator;
    var net = try build_modular_network(allocator);
    defer net.deinit();
    var c_nodes = net.get_control_nodes();
    try std.testing.expect(c_nodes.len == net.control_nodes.len);
}

test "Modular Network base nodes" {
    const allocator = std.testing.allocator;
    var net = try build_modular_network(allocator);
    defer net.deinit();
    var base_nodes = net.get_base_nodes();
    try std.testing.expect(base_nodes.len == net.all_nodes.len);
}
