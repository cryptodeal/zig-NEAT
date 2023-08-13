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

/// Network is a collection of all nodes within an organism's phenotype, which
/// effectively defines Neural Network topology. The point of the network is to
/// define a single entity which can evolve or learn on its own, even though
/// it may be part of a larger framework.
pub const Network = struct {
    /// The Networks's id
    id: i64,
    /// The Networks's name
    name: []const u8 = undefined,
    // Slice of NNodes that output from the Network.
    outputs: []*NNode,

    /// The number of links in the Network (0 if not yet counted).
    num_links: usize = 0,

    /// Slice of all NNodes in the Network (excluding MIMO control nodes).
    all_nodes: []*NNode,

    /// Slice of NNodes that input into the Network.
    inputs: []*NNode,

    /// Slice of NNodes that connect Network modules.
    control_nodes: []*NNode = undefined,
    /// Flag indicating whether Network control_nodes have been allocated (used for deinit).
    has_control_nodes: bool = false,

    /// List of all NNodes in the Network including MIMOControlNodes.
    all_nodes_MIMO: std.ArrayList(*NNode),

    /// The Network's associated Graph.
    graph: Graph(i64, void, f64) = undefined,

    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new Network.
    pub fn init(allocator: std.mem.Allocator, in: []*NNode, out: []*NNode, all: []*NNode, net_id: i64) !*Network {
        var self: *Network = try allocator.create(Network);
        var graph = Graph(i64, void, f64).init(allocator);
        for (all) |node| {
            for (node.outgoing.items) |edge| {
                try graph.addEdge(edge.in_node.?.id, {}, edge.out_node.?.id, {}, edge.cxn_weight);
            }
        }
        self.* = .{
            .allocator = allocator,
            .id = net_id,
            .inputs = in,
            .outputs = out,
            .all_nodes = all,
            .graph = graph,
            .all_nodes_MIMO = std.ArrayList(*NNode).init(allocator),
        };

        try self.all_nodes_MIMO.appendSlice(all);
        return self;
    }

    /// Initializes a new modular Network with control nodes.
    pub fn initModular(allocator: std.mem.Allocator, in: []*NNode, out: []*NNode, all: []*NNode, control: []*NNode, net_id: i64) !*Network {
        var self: *Network = try Network.init(allocator, in, out, all, net_id);
        self.control_nodes = control;
        self.has_control_nodes = true;
        for (control) |node| {
            for (node.incoming.items) |edge| {
                try self.graph.addEdge(edge.in_node.?.id, {}, edge.out_node.?.id, {}, edge.cxn_weight);
            }
            for (node.outgoing.items) |edge| {
                try self.graph.addEdge(edge.in_node.?.id, {}, edge.out_node.?.id, {}, edge.cxn_weight);
            }
        }
        try self.all_nodes_MIMO.appendSlice(control);
        return self;
    }

    /// Frees all associated memory.
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

    /// Returns a new NetworkSolver (implements Solver), backed by this Network instance.
    pub fn getSolver(self: *Network, allocator: std.mem.Allocator) !*NetworkSolver {
        return NetworkSolver.init(allocator, self);
    }

    /// Initializes a new FastNetworkSolver based on the architecture of this network.
    /// It's primarily aimed for big networks to improve processing speed.
    pub fn fastNetworkSolver(self: *Network, allocator: std.mem.Allocator) !Solver {
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
        var neuron_idx = try self.processList(0, bias_list.items, activations, &neuron_lookup);
        neuron_idx = try self.processList(neuron_idx, in_list.items, activations, &neuron_lookup);
        neuron_idx = try self.processList(neuron_idx, self.outputs, activations, &neuron_lookup);
        _ = try self.processList(neuron_idx, hidden_list.items, activations, &neuron_lookup);

        // walk through neurons in order: input, output, hidden and create bias and connections lists
        var biases = try allocator.alloc(f64, total_neuron_count);
        for (biases) |*v| v.* = 0;

        var connections = std.ArrayList(*FastNetworkLink).init(allocator);

        var in_connects = try self.processIncomingConnections(allocator, in_list.items, biases, &neuron_lookup);
        try connections.appendSlice(in_connects);
        allocator.free(in_connects);
        in_connects = try self.processIncomingConnections(allocator, hidden_list.items, biases, &neuron_lookup);
        try connections.appendSlice(in_connects);
        allocator.free(in_connects);
        in_connects = try self.processIncomingConnections(allocator, self.outputs, biases, &neuron_lookup);
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

    /// Unique Id generator for Network's Nodes.
    pub fn nodeIdGenerator(self: *Network) i64 {
        return @as(i64, @intCast(self.all_nodes.len));
    }

    fn processList(_: *Network, start_idx: usize, n_list: []*NNode, activations: []net_math.NodeActivationType, neuron_lookup: *std.AutoHashMap(i64, usize)) !usize {
        var idx = start_idx;
        for (n_list) |ne| {
            activations[idx] = ne.activation_type;
            try neuron_lookup.put(ne.id, idx);
            idx += 1;
        }
        return idx;
    }

    fn processIncomingConnections(_: *Network, allocator: std.mem.Allocator, n_list: []*NNode, biases: []f64, neuron_lookup: *std.AutoHashMap(i64, usize)) ![]*FastNetworkLink {
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

    /// Checks whether NNode with given Id is a control node.
    pub fn isControlNode(self: *Network, nid: i64) bool {
        if (self.has_control_nodes) {
            for (self.control_nodes) |cn| if (cn.id == nid) return true;
        }
        return false;
    }

    /// Flushes Network state by removing all current activations. Returns true if network
    /// flushed successfully; else returns error.
    pub fn flush(self: *Network) !bool {
        // flush back recursively
        for (self.all_nodes) |node| {
            node.flushback();
            try node.flushbackCheck();
        }
        return true;
    }

    // TODO: rework so printActivation accepts a writer as a parameter and writes to it.
    pub fn printActivation(self: *Network, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().print("Network {s} with id {d} outputs: (", .{ self.name, self.id });
        for (self.outputs, 0..) |node, i| {
            try buffer.writer().print("[Output #{d}: {any}] ", .{ i, node });
        }
        try buffer.writer().print(")", .{});
        return buffer.toOwnedSlice();
    }

    // TODO: rework so printInput accepts a writer as a parameter and writes to it.
    pub fn printInput(self: *Network, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().print("Network {s} with id {d} inputs: (", .{ self.name, self.id });
        for (self.inputs, 0..) |node, i| {
            try buffer.writer().print("[Input #{d}: {any}] ", .{ i, node });
        }
        try buffer.writer().print(")", .{});
        return buffer.toOwnedSlice();
    }

    /// If at least one output is not active, then return true; else return false.
    pub fn outputIsOff(self: *Network) bool {
        for (self.outputs) |node| {
            if (node.activations_count == 0) {
                return true;
            }
        }
        return false;
    }

    /// Attempts to activate the Network given number of steps before returning error.
    /// Normally the maxSteps should be equal to the maximal activation depth of the
    /// Network as returned by `maxActivationDepth` or `maxActivationDepthCapped`.
    pub fn activateSteps(self: *Network, max_steps: i64) !bool {
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
        while (self.outputIsOff() or !one_time) {
            if (abort_count >= max_steps) {
                return error.ErrNetExceededMaxActivationAttempts;
            }

            for (self.all_nodes) |np| {
                if (np.isNeuron()) {
                    // reset activation value
                    np.activation_sum = 0.0;

                    // For each node's incoming connection, add the activity from the connection to the activesum
                    for (np.incoming.items) |link| {
                        // handle potential time delayed cxns
                        if (!link.is_time_delayed) {
                            add_amount = link.cxn_weight * link.in_node.?.getActiveOut();
                            if (link.in_node.?.is_active or link.in_node.?.isSensor()) {
                                np.is_active = true;
                            }
                        } else {
                            add_amount = link.cxn_weight * link.in_node.?.getActiveOutTd();
                        }
                        np.activation_sum += add_amount;
                    } // End {for} over incoming links
                } // End if != SENSOR
            } // End {for} over all nodes

            // activate all the neuron nodes off their incoming activation
            // only activate if some active input recvd
            for (self.all_nodes) |np| {
                if (np.isNeuron()) {
                    // Only activate if some active input came in
                    if (np.is_active) {
                        try common.activateNode(np);
                    }
                }
            }

            // activate all MIMO control genes to propagate activation through genome modules
            if (self.has_control_nodes) {
                for (self.control_nodes) |cn| {
                    cn.is_active = false;
                    // activate control MIMO node as control module
                    try common.activateModule(cn);
                    cn.is_active = true;
                }
            }

            one_time = true;
            abort_count += 1;
        }

        return true;
    }

    /// Activate the network such that all outputs are active.
    pub fn activate(self: *Network) !bool {
        return self.activateSteps(20);
    }

    /// Propagates activation wave through all Network nodes provided number of steps in forward direction.
    /// Returns true if activation wave passed from all inputs to the outputs.
    pub fn forwardSteps(self: *Network, steps: i64) !bool {
        if (steps == 0) {
            return error.ErrZeroActivationStepsRequested;
        }
        var i: usize = 0;
        while (i < steps) : (i += 1) {
            _ = try self.activateSteps(steps);
        }
        return true;
    }

    /// Propagates activation wave through all network nodes provided number of steps by recursion from output nodes
    /// Returns true if activation wave passed from all inputs to the outputs. This method is preferred method
    /// of network activation when number of forward steps can not be easy calculated and no network modules are set.
    pub fn recursiveSteps(self: *Network) !bool {
        var net_depth = try self.maxActivationDepthCapped(0);
        return self.forwardSteps(net_depth);
    }

    /// Not yet implemented.
    pub fn relax(self: *Network) !bool {
        _ = self;
        std.debug.print("relax not implemented\n", .{});
        return error.ErrNotImplemented;
    }

    /// Set sensors values to the input (and bias) nodes of the Network.
    pub fn loadSensors(self: *Network, sensors: []f64) void {
        var counter: usize = 0;
        if (sensors.len == self.inputs.len) {
            // BIAS value provided as input
            for (self.inputs) |node| {
                if (node.isSensor()) {
                    _ = node.sensorLoad(sensors[counter]);
                    counter += 1;
                }
            }
        } else {
            // use default BIAS value
            for (self.inputs) |node| {
                if (node.neuron_type == NodeNeuronType.InputNeuron) {
                    _ = node.sensorLoad(sensors[counter]);
                    counter += 1;
                } else {
                    // default BIAS value
                    _ = node.sensorLoad(1.0);
                }
            }
        }
    }

    /// Read output values from the output nodes of the Network.
    pub fn readOutputs(self: *Network, allocator: std.mem.Allocator) ![]f64 {
        var outs = try allocator.alloc(f64, self.outputs.len);
        for (self.outputs, 0..) |o, i| {
            outs[i] = o.activation;
        }
        return outs;
    }

    /// Returns the number of nodes in the Network.
    pub fn nodeCount(self: *Network) usize {
        if (!self.has_control_nodes or self.control_nodes.len == 0) {
            return self.all_nodes.len;
        } else {
            return self.all_nodes.len + self.control_nodes.len;
        }
    }

    /// Returns the number of links in the Network.
    pub fn linkCount(self: *Network) usize {
        self.num_links = 0;
        for (self.all_nodes) |node| {
            self.num_links += node.incoming.items.len;
        }
        if (self.has_control_nodes and self.control_nodes.len != 0) {
            for (self.control_nodes) |node| {
                self.num_links += node.incoming.items.len;
                self.num_links += node.outgoing.items.len;
            }
        }
        return self.num_links;
    }

    /// Returns complexity of this Network, which is sum of nodes count and links count.
    pub fn complexity(self: *Network) usize {
        return self.nodeCount() + self.linkCount();
    }

    /// This checks a POTENTIAL link between a potential in_node
    /// and potential out_node to see if it must be recurrent.
    /// Use count and thresh to jump out in the case of an infinite loop.
    pub fn isRecurrent(self: *Network, in_node: *NNode, out_node: *NNode, count: *i64, thresh: i64) bool {
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
                    if (self.isRecurrent(link.in_node.?, out_node, count, thresh)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /// Used to find the maximum number of neuron layers to be activated between an output and an input layers.
    pub fn maxActivationDepth(self: *Network) !i64 {
        if (!self.has_control_nodes or self.control_nodes.len == 0) {
            return self.maxActivationDepthCapped(0);
        } else {
            return self.maxActivationDepthModular();
        }
    }

    /// Used to find the maximum number of neuron layers to be activated between an output and an input layers.
    /// It is possible to limit the maximal depth value by setting the `max_depth_cap` value greater than zero.
    /// If Network depth exceeds provided `max_depth_cap` value, returns error indicating that calculation stopped.
    /// If `max_depth_cap` is less or equal to zero no maximal depth limitation will be set. Unsupported for modular Networks.
    pub fn maxActivationDepthCapped(self: *Network, max_depth_cap: i64) !i64 {
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

    /// Returns all Network nodes including MIMO control nodes (base nodes + control nodes).
    pub fn getAllNodes(self: *Network) []*NNode {
        return self.all_nodes_MIMO.items;
    }

    /// Returns all control nodes of this Network.
    pub fn getControlNodes(self: *Network) []*NNode {
        return self.control_nodes;
    }

    /// Returns all nodes in this Network (excluding MIMO control nodes).
    pub fn getBaseNodes(self: *Network) []*NNode {
        return self.all_nodes;
    }

    /// Calculates maximal activation depth and optionally prints the examined activation paths
    /// to the provided writer. It is intended only for modular Networks.
    pub fn maxActivationDepthModular(self: *Network) !i64 {
        var all_paths = self.graph.johnsonAllPaths() catch try self.graph.floydWarshall();
        defer all_paths.deinit();
        var max: usize = 0;
        for (self.inputs) |in| {
            for (self.outputs) |out| {
                var paths = try all_paths.allBetween(in.id, out.id);
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

/// NetworkSolver implements Solver for a given Network.
pub const NetworkSolver = struct {
    /// The Network backing this NetworkSolver.
    network: *Network,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new NetworkSolver.
    pub fn init(allocator: std.mem.Allocator, network: *Network) !*NetworkSolver {
        var self = try allocator.create(NetworkSolver);
        self.* = .{
            .allocator = allocator,
            .network = network,
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *NetworkSolver) void {
        self.allocator.destroy(self);
    }

    /// Implements `forwardSteps` for Solver; calls into `Network.forwardSteps`.
    pub fn forwardSteps(self: *NetworkSolver, allocator: std.mem.Allocator, steps: usize) !bool {
        _ = allocator;
        return self.network.forwardSteps(@as(i64, @intCast(steps)));
    }

    /// Implements `recursiveSteps` for Solver; calls into `Network.recursiveSteps`.
    pub fn recursiveSteps(self: *NetworkSolver) !bool {
        return self.network.recursiveSteps();
    }

    /// Implements `relax` for Solver; calls into `Network.relax`, which is
    /// not yet implemented and will always return an error.
    pub fn relax(self: *NetworkSolver, allocator: std.mem.Allocator, max_steps: usize, max_allowed_signal_delta: f64) !bool {
        _ = allocator;
        _ = max_steps;
        _ = max_allowed_signal_delta;
        return self.network.relax();
    }

    /// Implements `flush` for Solver; calls into `Network.flush`.
    pub fn flush(self: *NetworkSolver) !bool {
        return self.network.flush();
    }

    /// Implements `loadSensors` for Solver; calls into `Network.loadSensors`.
    pub fn loadSensors(self: *NetworkSolver, sensors: []f64) !void {
        return self.network.loadSensors(sensors);
    }

    /// Implements `readOutputs` for Solver; calls into `Network.readOutputs`.
    pub fn readOutputs(self: *NetworkSolver, allocator: std.mem.Allocator) ![]f64 {
        return self.network.readOutputs(allocator);
    }

    /// Implements `nodeCount` for Solver; calls into `Network.nodeCount`.
    pub fn nodeCount(self: *NetworkSolver) usize {
        return self.network.nodeCount();
    }

    /// Implements `linkCount` for Solver; calls into `Network.linkCount`.
    pub fn linkCount(self: *NetworkSolver) usize {
        return self.network.linkCount();
    }
};

// test utils
fn buildPlainNetwork(allocator: std.mem.Allocator) !*Network {
    var all_nodes = try allocator.alloc(*NNode, 5);
    all_nodes[0] = try NNode.init(allocator, 1, .InputNeuron);
    all_nodes[1] = try NNode.init(allocator, 2, .InputNeuron);
    all_nodes[2] = try NNode.init(allocator, 3, .BiasNeuron);
    all_nodes[3] = try NNode.init(allocator, 7, .OutputNeuron);
    all_nodes[4] = try NNode.init(allocator, 8, .OutputNeuron);

    // OUTPUT 7
    _ = try all_nodes[3].connectFrom(allocator, all_nodes[1], 7);
    _ = try all_nodes[3].connectFrom(allocator, all_nodes[2], 4.5);
    // OUTPUT 8
    _ = try all_nodes[4].connectFrom(allocator, all_nodes[3], 13);

    var in = try allocator.alloc(*NNode, 3);
    @memcpy(in, all_nodes[0..3]);
    var out = try allocator.alloc(*NNode, 2);
    @memcpy(out, all_nodes[3..5]);

    return Network.init(allocator, in, out, all_nodes, 0);
}

fn buildDisconnectedNetwork(allocator: std.mem.Allocator) !*Network {
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

pub fn buildNetwork(allocator: std.mem.Allocator) !*Network {
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
    _ = try all_nodes[3].connectFrom(allocator, all_nodes[0], 15);
    _ = try all_nodes[3].connectFrom(allocator, all_nodes[1], 10);
    // HIDDEN 5
    _ = try all_nodes[4].connectFrom(allocator, all_nodes[1], 5);
    _ = try all_nodes[4].connectFrom(allocator, all_nodes[2], 1);
    // HIDDEN 6
    _ = try all_nodes[5].connectFrom(allocator, all_nodes[4], 17);
    // OUTPUT 7
    _ = try all_nodes[6].connectFrom(allocator, all_nodes[3], 7);
    _ = try all_nodes[6].connectFrom(allocator, all_nodes[5], 4.5);
    // OUTPUT 8
    _ = try all_nodes[7].connectFrom(allocator, all_nodes[5], 13);

    var in = try allocator.alloc(*NNode, 3);
    @memcpy(in, all_nodes[0..3]);
    var out = try allocator.alloc(*NNode, 2);
    @memcpy(out, all_nodes[6..8]);

    return Network.init(allocator, in, out, all_nodes, 0);
}

pub fn buildModularNetwork(allocator: std.mem.Allocator) !*Network {
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
    _ = try control_nodes[0].addIncoming(allocator, all_nodes[3], 1); // <- HIDDEN 4
    _ = try control_nodes[0].addIncoming(allocator, all_nodes[4], 1); // <- HIDDEN 5
    _ = try control_nodes[0].addOutgoing(allocator, all_nodes[5], 1); // <- HIDDEN 5

    // HIDDEN 4
    all_nodes[3].activation_type = .LinearActivation;
    _ = try all_nodes[3].connectFrom(allocator, all_nodes[0], 15); // <- INPUT 1
    _ = try all_nodes[3].connectFrom(allocator, all_nodes[2], 10); // <- BIAS
    // HIDDEN 5
    all_nodes[4].activation_type = .LinearActivation;
    _ = try all_nodes[4].connectFrom(allocator, all_nodes[1], 5); // <- INPUT 2
    _ = try all_nodes[4].connectFrom(allocator, all_nodes[2], 1); // <- BIAS

    // HIDDEN 7
    all_nodes[5].activation_type = .NullActivation;

    // OUTPUT 8
    all_nodes[6].activation_type = .LinearActivation;
    _ = try all_nodes[6].connectFrom(allocator, all_nodes[5], 4.5); // <- HIDDEN 7
    // OUTPUT 9
    all_nodes[7].activation_type = .LinearActivation;
    _ = try all_nodes[7].connectFrom(allocator, all_nodes[5], 13); // <- HIDDEN 7

    var in = try allocator.alloc(*NNode, 3);
    @memcpy(in, all_nodes[0..3]);
    var out = try allocator.alloc(*NNode, 2);
    @memcpy(out, all_nodes[6..8]);
    return Network.initModular(allocator, in, out, all_nodes, control_nodes, 0);
}

test "Modular Network activate" {
    var net = try buildModularNetwork(std.testing.allocator);
    defer net.deinit();
    var data = [_]f64{ 1, 2, 1 };
    net.loadSensors(&data);
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        var res = try net.activate();
        try std.testing.expect(res);
    }

    try std.testing.expect(net.outputs[0].activation == 1237.5);
    try std.testing.expect(net.outputs[1].activation == 3575.0);
}

test "Network max activation depth (simple)" {
    var net = try buildNetwork(std.testing.allocator);
    defer net.deinit();
    var depth = try net.maxActivationDepth();
    try std.testing.expect(depth == 3);

    // TODO: log network activation path
}

test "Modular Network max activation depth" {
    var net = try buildModularNetwork(std.testing.allocator);
    defer net.deinit();
    var depth = try net.maxActivationDepth();
    try std.testing.expect(depth == 4);

    // TODO: log network activation path
}

test "Network max activation depth (no hidden or control)" {
    var net = try buildPlainNetwork(std.testing.allocator);
    defer net.deinit();
    var depth = try net.maxActivationDepth();
    try std.testing.expect(depth == 1);
}

test "Network max activation depth capped (simple)" {
    var net = try buildNetwork(std.testing.allocator);
    defer net.deinit();
    var depth = try net.maxActivationDepthCapped(0);
    try std.testing.expect(depth == 3);

    // TODO: log network activation path

}

test "Network max activation depth (simple - exceed cap)" {
    var net = try buildNetwork(std.testing.allocator);
    defer net.deinit();
    var depth = try net.maxActivationDepthCapped(2);
    try std.testing.expect(depth == 2);
}

test "Modular Network max activation depth capped" {
    var net = try buildModularNetwork(std.testing.allocator);
    defer net.deinit();
    try std.testing.expectError(error.ErrModularNetworkUnsupported, net.maxActivationDepthCapped(0));
}

test "Network output is off" {
    var net = try buildNetwork(std.testing.allocator);
    defer net.deinit();
    try std.testing.expect(net.outputIsOff());
}

test "Network activate" {
    var net = try buildNetwork(std.testing.allocator);
    defer net.deinit();

    var res = try net.activate();
    try std.testing.expect(res);

    // check activation
    for (net.all_nodes) |node| {
        if (node.isNeuron()) {
            try std.testing.expect(node.activations_count != 0);
            try std.testing.expect(node.activation != 0);

            // Check activation and time delayed activation
            try std.testing.expect(node.getActiveOut() != 0);
        }
    }
}

test "Network forward steps" {
    const allocator = std.testing.allocator;
    var net = try buildNetwork(allocator);
    defer net.deinit();

    var res = try net.forwardSteps(3);
    try std.testing.expect(res);

    var expected_outs = [_]f64{ 1, 1 };
    var net_outs = try net.readOutputs(allocator);
    defer allocator.free(net_outs);
    try std.testing.expect(net_outs.len == expected_outs.len);
    try std.testing.expectEqualSlices(f64, &expected_outs, net_outs);

    // test zero steps
    try std.testing.expectError(error.ErrZeroActivationStepsRequested, net.forwardSteps(0));
}

test "Network recursive steps" {
    const allocator = std.testing.allocator;
    var net = try buildNetwork(allocator);
    defer net.deinit();

    var data = [_]f64{ 0.5, 0, 1.5 };
    net.loadSensors(&data);

    var relaxed = try net.recursiveSteps();
    try std.testing.expect(relaxed);

    // TODO: log network activation path
}

test "Network forward steps (disconnected)" {
    const allocator = std.testing.allocator;
    var net = try buildDisconnectedNetwork(allocator);
    defer net.deinit();
    try std.testing.expectError(error.ErrNetExceededMaxActivationAttempts, net.forwardSteps(3));
}

test "Network load sensors" {
    const allocator = std.testing.allocator;
    var net = try buildNetwork(allocator);
    defer net.deinit();

    var sensors = [_]f64{ 1, 3.4, 5.6 };
    net.loadSensors(&sensors);
    var counter: usize = 0;
    for (net.getAllNodes()) |node| {
        if (node.isSensor()) {
            try std.testing.expect(sensors[counter] == node.activation);
            try std.testing.expect(node.activations_count == 1);
            counter += 1;
        }
    }
}

test "Network flush" {
    const allocator = std.testing.allocator;
    var net = try buildNetwork(allocator);
    defer net.deinit();

    // activate and check state
    var res = try net.activate();
    try std.testing.expect(res);

    // flush and check
    res = try net.flush();
    try std.testing.expect(res);

    for (net.getAllNodes()) |node| {
        try std.testing.expect(node.activations_count == 0);
        try std.testing.expect(node.activation == 0);

        // Check activation and time delayed activation
        try std.testing.expect(node.getActiveOut() == 0);
        try std.testing.expect(node.getActiveOutTd() == 0);
    }
}

test "Network node count" {
    const allocator = std.testing.allocator;
    var net = try buildNetwork(allocator);
    defer net.deinit();

    try std.testing.expect(net.nodeCount() == 8);
}

test "Network link count" {
    const allocator = std.testing.allocator;
    var net = try buildNetwork(allocator);
    defer net.deinit();

    try std.testing.expect(net.linkCount() == 8);
}

test "Network fast network solver" {
    const allocator = std.testing.allocator;
    var net = try buildNetwork(allocator);
    defer net.deinit();

    var solver = try net.fastNetworkSolver(allocator);
    defer solver.deinit();

    // check solver structure
    try std.testing.expect(solver.nodeCount() == @as(usize, @intCast(net.nodeCount())));
    try std.testing.expect(solver.linkCount() == @as(usize, @intCast(net.linkCount())));
}

test "Network activate steps with zero activation steps" {
    const allocator = std.testing.allocator;
    var net = try buildNetwork(allocator);
    defer net.deinit();
    try std.testing.expectError(error.ErrZeroActivationStepsRequested, net.activateSteps(0));
}

test "Network activate steps exceed max activation attempts" {
    const allocator = std.testing.allocator;
    var net = try buildDisconnectedNetwork(allocator);
    defer net.deinit();
    try std.testing.expectError(error.ErrNetExceededMaxActivationAttempts, net.activateSteps(3));
}

test "Modular Network control nodes" {
    const allocator = std.testing.allocator;
    var net = try buildModularNetwork(allocator);
    defer net.deinit();
    var c_nodes = net.getControlNodes();
    try std.testing.expect(c_nodes.len == net.control_nodes.len);
}

test "Modular Network base nodes" {
    const allocator = std.testing.allocator;
    var net = try buildModularNetwork(allocator);
    defer net.deinit();
    var base_nodes = net.getBaseNodes();
    try std.testing.expect(base_nodes.len == net.all_nodes.len);
}
