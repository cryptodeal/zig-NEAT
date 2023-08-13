const std = @import("std");
const net_math = @import("../math/activations.zig");
const net = @import("network.zig");

const buildNetwork = net.buildNetwork;
const buildModularNetwork = net.buildModularNetwork;

/// FastNetworkLink The connection descriptor for fast network.
pub const FastNetworkLink = struct {
    /// The index of source neuron.
    source_idx: usize,
    /// The index of target neuron.
    target_idx: usize,
    /// The link's weight.
    weight: f64 = 0,
    /// The signal relayed by this link.
    signal: f64 = 0,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Intializes a new FastNetworkLink.
    pub fn init(allocator: std.mem.Allocator, source: usize, target: usize, weight: f64) !*FastNetworkLink {
        var self = try allocator.create(FastNetworkLink);
        self.* = .{
            .allocator = allocator,
            .source_idx = source,
            .target_idx = target,
            .weight = weight,
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *FastNetworkLink) void {
        self.allocator.destroy(self);
    }
};

/// FastControlNode The module relay (control node) descriptor for fast network.
pub const FastControlNode = struct {
    // The activation function for control node.
    activation_type: net_math.NodeActivationType,
    // The indices of input nodes.
    input_idxs: []usize,
    // The indices of output nodes.
    output_idxs: []usize,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new FastControlNode.
    pub fn init(allocator: std.mem.Allocator, inputs: []usize, outputs: []usize, activation: net_math.NodeActivationType) !*FastControlNode {
        var self = try allocator.create(FastControlNode);
        self.* = .{
            .allocator = allocator,
            .input_idxs = inputs,
            .output_idxs = outputs,
            .activation_type = activation,
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *FastControlNode) void {
        self.allocator.free(self.input_idxs);
        self.allocator.free(self.output_idxs);
        self.allocator.destroy(self);
    }
};

/// FastModularNetworkSolver is the network solver implementation to be used for large neural networks simulation.
pub const FastModularNetworkSolver = struct {
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// The network's id.
    id: i64 = undefined,
    /// The network's name.
    name: []const u8 = undefined,
    /// The current activation values per each neuron.
    neuron_signals: []f64,
    /// This parallels `neuron_signals` and is used to test network relaxation.
    neuron_signals_processing: []f64,

    /// The activation functions per neuron, must be ordered same as `neuron_signals`
    /// has undefined entries for neurons that are inputs/outputs of module
    activation_functions: []net_math.NodeActivationType,
    // The bias values associated with neurons.
    bias_list: ?[]f64,
    /// The control nodes relaying between network modules.
    modules: ?[]*FastControlNode,
    /// The network's connections.
    cxns: []*FastNetworkLink,

    /// Count of the network's input neurons.
    input_neuron_count: usize,
    /// The total count of sensors in the network (input + bias); also marks index of the first output neuron in neuron signals.
    sensor_neuron_count: usize,
    /// Count of the network's output neurons.
    output_neuron_count: usize,
    /// Count of the network's bias neurons (usually 1); also marks index of the first input neuron in neuron signals.
    bias_neuron_count: usize,
    /// The total count of neurons in the network.
    total_neuron_count: usize,

    /// For recursive activation, tracks whether node has been activated.
    activated: []bool,
    /// For recursive activation, tracks whether node is currently being calculated (recurrent cxns processing).
    in_activation: []bool,
    /// For recursive activation, tracks previous activation values of recurrent cxns (recurrent cxns processing)
    last_activation: []f64,

    /// Holds IDs of outgoing nodes for each network node.
    adjacent_list: []std.ArrayList(usize),
    /// Holds IDs of incoming nodes for each network node.
    reverse_adjacent_list: []std.ArrayList(usize),
    /// Holds cxn weights from all connected nodes.
    adjacent_matrix: [][]f64,

    /// Initializes a new FastModularNetworkSolver.
    pub fn init(allocator: std.mem.Allocator, bias_neuron_count: usize, input_neuron_count: usize, output_neuron_count: usize, total_neuron_count: usize, activation_fns: []net_math.NodeActivationType, cxns: []*FastNetworkLink, bias_list: ?[]f64, modules: ?[]*FastControlNode) !*FastModularNetworkSolver {
        // Allocate the arrays that store the states at different points in the neural network.
        // The neuron signals are initialised to 0 by default. Only bias nodes need setting to 1.
        var neuron_signals = try allocator.alloc(f64, total_neuron_count);
        var neuron_signals_processing = try allocator.alloc(f64, total_neuron_count);

        // Allocate activation arrays
        var activated = try allocator.alloc(bool, total_neuron_count);
        var in_activation = try allocator.alloc(bool, total_neuron_count);
        var last_activation = try allocator.alloc(f64, total_neuron_count);

        // alloc adjacent lists/matrix for fast access to incoming/outgoing nodes & cxn weights
        var adjacent_list = try allocator.alloc(std.ArrayList(usize), total_neuron_count);
        var reverse_adjacent_list = try allocator.alloc(std.ArrayList(usize), total_neuron_count);
        var adjacent_matrix = try allocator.alloc([]f64, total_neuron_count);

        // set initial values in single pass
        var i: usize = 0;
        while (i < total_neuron_count) : (i += 1) {
            neuron_signals[i] = if (i < bias_neuron_count) 1 else 0;
            neuron_signals_processing[i] = 0;
            activated[i] = false;
            in_activation[i] = false;
            last_activation[i] = 0;
            adjacent_list[i] = std.ArrayList(usize).init(allocator);
            reverse_adjacent_list[i] = std.ArrayList(usize).init(allocator);
            adjacent_matrix[i] = try allocator.alloc(f64, total_neuron_count);
        }

        var self = try allocator.create(FastModularNetworkSolver);
        self.* = .{
            .allocator = allocator,
            .bias_neuron_count = bias_neuron_count,
            .input_neuron_count = input_neuron_count,
            .sensor_neuron_count = bias_neuron_count + input_neuron_count,
            .output_neuron_count = output_neuron_count,
            .total_neuron_count = total_neuron_count,
            .activation_functions = activation_fns,
            .bias_list = bias_list,
            .modules = modules,
            .cxns = cxns,
            .neuron_signals = neuron_signals,
            .neuron_signals_processing = neuron_signals_processing,
            .activated = activated,
            .in_activation = in_activation,
            .last_activation = last_activation,
            .adjacent_list = adjacent_list,
            .reverse_adjacent_list = reverse_adjacent_list,
            .adjacent_matrix = adjacent_matrix,
        };

        for (cxns, 0..) |c, idx| {
            var crs = c.source_idx;
            var crt = c.target_idx;

            // holds outgoing nodes
            try self.adjacent_list[crs].append(crt);
            // holds incoming nodes
            try self.reverse_adjacent_list[crt].append(crs);
            // holds link weight
            self.adjacent_matrix[crs][crt] = cxns[idx].weight;
        }

        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *FastModularNetworkSolver) void {
        self.allocator.free(self.neuron_signals);
        self.allocator.free(self.neuron_signals_processing);
        self.allocator.free(self.activated);
        self.allocator.free(self.in_activation);
        self.allocator.free(self.last_activation);
        // free adjacency list
        for (self.adjacent_list) |v| v.deinit();
        self.allocator.free(self.adjacent_list);
        // free reverse adjacency list
        for (self.reverse_adjacent_list) |v| v.deinit();
        self.allocator.free(self.reverse_adjacent_list);
        // free adjacency matrix
        for (self.adjacent_matrix) |v| self.allocator.free(v);
        self.allocator.free(self.adjacent_matrix);
        // free activation fns
        self.allocator.free(self.activation_functions);
        // free cxn list
        for (self.cxns) |c| c.deinit();
        self.allocator.free(self.cxns);
        // free bias list
        if (self.bias_list != null) {
            self.allocator.free(self.bias_list.?);
        }
        // free control nodes
        if (self.modules != null) {
            for (self.modules.?) |m| m.deinit();
            self.allocator.free(self.modules.?);
        }
        self.allocator.destroy(self);
    }

    /// Propagates activation wave through all network nodes provided number of steps in forward direction.
    /// Returns true if activation wave passed from all inputs to the outputs.
    pub fn forwardSteps(self: *FastModularNetworkSolver, allocator: std.mem.Allocator, steps: usize) !bool {
        var res: bool = undefined;
        var i: usize = 0;
        while (i < steps) : (i += 1) {
            res = try self.forwardStep(allocator, 0);
        }
        return res;
    }

    /// Propagates activation wave through all network nodes provided number of steps by recursion from output nodes
    /// Returns true if activation wave passed from all inputs to the outputs. This method is preferred method
    /// of network activation when number of forward steps can not be easy calculated and no network modules are set.
    pub fn recursiveSteps(self: *FastModularNetworkSolver) !bool {
        if (self.modules != null and self.modules.?.len > 0) {
            std.debug.print("recursive activation can not be used for network with defined modules", .{});
            return error.FailedRecursiveStep;
        }

        var res: bool = undefined;

        var i: usize = 0;
        while (i < self.total_neuron_count) : (i += 1) {
            // if i is input node, set activated (true); else set not activated (false)
            self.activated[i] = i < self.sensor_neuron_count;
            self.in_activation[i] = false;
            // set last activation for output/hidden neurons
            if (i >= self.sensor_neuron_count) {
                self.last_activation[i] = self.neuron_signals[i];
            }
        }

        i = 0;
        while (i < self.output_neuron_count) : (i += 1) {
            const index = self.sensor_neuron_count + i;
            // TODO: below function call is not implemented
            res = try self.recursiveActivateNode(index);
            if (!res) {
                std.debug.print("failed to recursively activate the output neuron at {d}", .{index});
                return error.FailedRecursiveStep;
            }
        }

        return res;
    }

    /// Propagate activation wave by recursively looking for input signals graph for a given output neuron.
    pub fn recursiveActivateNode(self: *FastModularNetworkSolver, current_node: usize) !bool {
        // return if we've reached an input node (signal already set)
        if (self.activated[current_node]) {
            self.in_activation[current_node] = false;
            return true;
        }

        // flag node as in process (currently being calculated)
        self.in_activation[current_node] = true;

        // set pre-signal to `0`
        self.neuron_signals_processing[current_node] = 0;

        // walk through adjacency list and activated
        var i: usize = 0;

        while (i < self.reverse_adjacent_list[current_node].items.len) : (i += 1) {
            var current_adj_node: usize = self.reverse_adjacent_list[current_node].items[i];
            // if node in calculation, reached cycle/recurrent cxn
            // use prev activation value
            if (self.in_activation[current_adj_node]) {
                self.neuron_signals_processing[current_node] += self.last_activation[current_adj_node] * self.adjacent_matrix[current_adj_node][current_node];
            } else {
                // else proceed normally
                // recurse if neuron not yet activated
                if (!self.activated[current_adj_node]) {
                    const res = try self.recursiveActivateNode(current_adj_node);
                    if (!res) {
                        std.debug.print("failed to recursively activate neuron at {d}", .{current_adj_node});
                        return false;
                    }
                }
                // add node to activated list
                self.neuron_signals_processing[current_node] += self.neuron_signals[current_adj_node] * self.adjacent_matrix[current_adj_node][current_node];
            }
        }

        // mark neuron as completed
        self.activated[current_node] = true;

        // neuron is no longer in process
        self.in_activation[current_node] = false;

        // set signal now that activation is complete
        self.neuron_signals[current_node] = try net_math.NodeActivationType.activateByType(self.neuron_signals_processing[current_node], null, self.activation_functions[current_node]);
        return true;
    }

    /// Attempts to relax network given amount of steps until giving up. The network considered relaxed when absolute
    /// value of the change at any given point is less than max_allowed_signal_delta during activation waves propagation.
    /// If max_allowed_signal_delta value is less than or equal to 0, the method will return true without checking for relaxation.
    pub fn relax(self: *FastModularNetworkSolver, allocator: std.mem.Allocator, max_steps: usize, max_allowed_signal_delta: f64) !bool {
        var relaxed: bool = false;
        var i: usize = 0;
        while (i < max_steps) : (i += 1) {
            relaxed = try self.forwardStep(allocator, max_allowed_signal_delta);
            if (relaxed) break;
        }
        return relaxed;
    }

    /// Performs single forward step through the network and tests if network become relaxed. The network considered relaxed
    /// when absolute value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
    pub fn forwardStep(self: *FastModularNetworkSolver, allocator: std.mem.Allocator, max_allowed_signal_delta: f64) !bool {
        var is_relaxed = true;

        // calculate output signal per each cxn and add signals to target neurons
        for (self.cxns) |c| {
            self.neuron_signals_processing[c.target_idx] += self.neuron_signals[c.source_idx] * c.weight;
        }

        // pass signals through single val activation functions
        var i: usize = self.sensor_neuron_count;
        while (i < self.total_neuron_count) : (i += 1) {
            var signal = self.neuron_signals_processing[i];

            if (self.bias_neuron_count > 0) {
                // append BIAS value if need be
                signal += self.bias_list.?[i];
            }
            self.neuron_signals_processing[i] = try net_math.NodeActivationType.activateByType(signal, undefined, self.activation_functions[i]);
        }

        // pass signals through each module (activation function with more than one input or output)
        for (self.modules.?) |module| {
            var inputs = try allocator.alloc(f64, module.input_idxs.len);
            defer allocator.free(inputs);
            for (module.input_idxs, 0..) |inIndex, idx| {
                inputs[idx] = self.neuron_signals_processing[inIndex];
            }
            var outputs = try net_math.NodeActivationType.activateModuleByType(inputs, undefined, module.activation_type);
            for (module.output_idxs, 0..) |outIndex, idx| {
                self.neuron_signals_processing[outIndex] = outputs[idx];
            }
        }

        // move all neuron signals changed in processing network activation to storage
        if (max_allowed_signal_delta <= 0) {
            // iterate through output/hidden neurons, collecting activations
            var idx: usize = self.sensor_neuron_count;
            while (idx < self.total_neuron_count) : (idx += 1) {
                self.neuron_signals[idx] = self.neuron_signals_processing[idx];
                self.neuron_signals_processing[idx] = 0;
            }
        } else {
            var idx: usize = self.sensor_neuron_count;
            while (idx < self.total_neuron_count) : (idx += 1) {
                // check if any location in the network has changed by more than a small amount
                is_relaxed = is_relaxed and !(@fabs(self.neuron_signals[idx] - self.neuron_signals_processing[idx]) > max_allowed_signal_delta);
                self.neuron_signals[idx] = self.neuron_signals_processing[idx];
                self.neuron_signals_processing[idx] = 0;
            }
        }

        return is_relaxed;
    }

    /// Flushes network state by removing all current activations. Returns true if network
    /// flushed successfully; else returns error.
    pub fn flush(self: *FastModularNetworkSolver) !bool {
        var i: usize = self.bias_neuron_count;
        while (i < self.total_neuron_count) : (i += 1) {
            self.neuron_signals[i] = 0.0;
        }
        return true;
    }

    /// Set sensors values to the input nodes of the network.
    pub fn loadSensors(self: *FastModularNetworkSolver, inputs: []f64) !void {
        if (inputs.len == self.input_neuron_count) {
            // only inputs should be provided
            var i: usize = 0;
            while (i < self.input_neuron_count) : (i += 1) {
                self.neuron_signals[self.bias_neuron_count + i] = inputs[i];
            }
        } else {
            std.debug.print("the sensors array size is unsupported by network solver\n", .{});
            return error.ErrNetUnsupportedSensorsArraySize;
        }
    }

    /// Read output values from the output nodes of the network.
    pub fn readOutputs(self: *FastModularNetworkSolver, allocator: std.mem.Allocator) ![]f64 {
        // decouple & return
        var outs = try allocator.alloc(f64, self.output_neuron_count);
        @memcpy(outs, self.neuron_signals[self.sensor_neuron_count .. self.sensor_neuron_count + self.output_neuron_count]);
        return outs;
    }

    /// Returns the total number of neural units in the network.
    pub fn nodeCount(self: *FastModularNetworkSolver) usize {
        return self.total_neuron_count + if (self.modules != null) self.modules.?.len else 0;
    }

    /// Returns the total number of links between nodes in the network.
    pub fn linkCount(self: *FastModularNetworkSolver) usize {
        // count all cxns
        var num_links: usize = self.cxns.len;

        // count all bias links if any
        if (self.bias_neuron_count > 0) {
            for (self.bias_list.?) |b| {
                if (b != 0) num_links += 1;
            }
        }

        // count all module links
        if (self.modules != null and self.modules.?.len != 0) {
            for (self.modules.?) |m| {
                num_links += m.input_idxs.len + m.output_idxs.len;
            }
        }
        return num_links;
    }

    /// Formats FastModularNetworkSolver for printing to writer.
    pub fn format(value: FastModularNetworkSolver, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("FastModularNetwork, id: {d}, name: [{s}], neurons: {d},\n\tinputs: {d},\tbias: {d},\toutputs: {d},\t hidden: {d}", .{ value.id, value.name, value.total_neuron_count, value.input_neuron_count, value.bias_neuron_count, value.output_neuron_count, (value.total_neuron_count - value.sensor_neuron_count - value.output_neuron_count) });
    }
};

fn countActiveSignals(impl: *FastModularNetworkSolver) usize {
    var active: usize = 0;
    var i: usize = impl.bias_neuron_count;
    while (i < impl.total_neuron_count) : (i += 1) {
        if (impl.neuron_signals[i] != 0) {
            active += 1;
        }
    }
    return active;
}

test "FastModularNetworkSolver load sensors" {
    const allocator = std.testing.allocator;
    var n = try buildNetwork(allocator);
    defer n.deinit();

    var fmm = try n.fastNetworkSolver(allocator);
    defer fmm.deinit();

    // test normal
    var d1 = [_]f64{ 0.5, 1.1 };
    var erred = false;
    fmm.loadSensors(&d1) catch {
        erred = true;
    };
    try std.testing.expect(!erred);

    var d2 = [_]f64{ 0.5, 1.1, 1 };
    try std.testing.expectError(error.ErrNetUnsupportedSensorsArraySize, fmm.loadSensors(&d2));
}

test "FastModularNetworkSolver recursive steps" {
    const allocator = std.testing.allocator;
    var n = try buildNetwork(allocator);
    defer n.deinit();

    // Create network solver
    var d1 = [_]f64{ 0.5, 1.1 };
    var fmm = try n.fastNetworkSolver(allocator);
    defer fmm.deinit();
    var erred = false;
    fmm.loadSensors(&d1) catch {
        erred = true;
    };
    try std.testing.expect(!erred);

    // Activate objective network
    var d2 = [_]f64{ 0.5, 1.1, 1 }; // BIAS is third value
    n.loadSensors(&d2);

    var depth = n.maxActivationDepth() catch blk: {
        erred = true;
        break :blk -1;
    };
    try std.testing.expect(!erred);

    std.debug.print("depth: {d}\n", .{depth});
    // TODO: log network activation path

    var res = try n.forwardSteps(depth);
    try std.testing.expect(res);

    // Do recursive activation of the Fast Network Solver
    res = try fmm.recursiveSteps();
    try std.testing.expect(res);

    // Compare activations of objective network and Fast Network Solver
    var fmm_outputs = try fmm.readOutputs(allocator);
    defer allocator.free(fmm_outputs);
    try std.testing.expect(fmm_outputs.len == n.outputs.len);
    for (fmm_outputs, 0..) |out, i| {
        try std.testing.expect(out == n.outputs[i].activation);
    }
}

test "FastModularNetworkSolver forward steps" {
    const allocator = std.testing.allocator;
    var n = try buildModularNetwork(allocator);
    defer n.deinit();

    // create network solver
    var d1 = [_]f64{ 1, 2 }; // bias inherent
    var fmm = try n.fastNetworkSolver(allocator);
    defer fmm.deinit();
    var erred = false;
    fmm.loadSensors(&d1) catch {
        erred = true;
    };
    try std.testing.expect(!erred);

    var depth = n.maxActivationDepth() catch blk: {
        erred = true;
        break :blk -1;
    };
    try std.testing.expect(!erred);

    std.debug.print("depth: {d}\n", .{depth});
    // TODO: log network activation path

    // activate objective network
    var d2 = [_]f64{ 1, 2, 1 }; // bias is third value
    n.loadSensors(&d2);
    var res = try n.forwardSteps(depth);
    try std.testing.expect(res);

    // do forward steps through the solver and test results
    res = try fmm.forwardSteps(allocator, @as(usize, @intCast(depth)));
    try std.testing.expect(res);

    // check results by comparing activations of objective network and fast network solver
    var fmm_outputs = try fmm.readOutputs(allocator);
    defer allocator.free(fmm_outputs);
    try std.testing.expect(fmm_outputs.len == n.outputs.len);
    for (fmm_outputs, 0..) |out, i| {
        try std.testing.expect(out == n.outputs[i].activation);
    }
}

test "FastModularNetworkSolver relax" {
    const allocator = std.testing.allocator;
    var n = try buildModularNetwork(allocator);
    defer n.deinit();

    // create network solver
    var d1 = [_]f64{ 1.5, 2 }; // bias inherent
    var fmm = try n.fastNetworkSolver(allocator);
    defer fmm.deinit();
    var erred = false;
    fmm.loadSensors(&d1) catch {
        erred = true;
    };
    try std.testing.expect(!erred);

    var depth = n.maxActivationDepth() catch blk: {
        erred = true;
        break :blk -1;
    };
    try std.testing.expect(!erred);

    std.debug.print("depth: {d}\n", .{depth});
    // TODO: log network activation path

    var d2 = [_]f64{ 1.5, 2, 1 }; // bias is third value
    n.loadSensors(&d2);

    var res = try n.forwardSteps(depth);
    try std.testing.expect(res);

    // relax fast network solver
    res = fmm.relax(allocator, @as(usize, @intCast(depth)), 1) catch blk: {
        erred = true;
        break :blk false;
    };
    try std.testing.expect(!erred);
    try std.testing.expect(res);

    // check results by comparing activations of objective network and fast network solver
    var fmm_outputs = try fmm.readOutputs(allocator);
    defer allocator.free(fmm_outputs);
    for (fmm_outputs, 0..) |out, i| {
        try std.testing.expect(out == n.outputs[i].activation);
    }
}

test "FastModularNetworkSolver flush" {
    const allocator = std.testing.allocator;
    var n = try buildModularNetwork(allocator);
    defer n.deinit();

    // create network solver
    var d1 = [_]f64{ 1.5, 2 }; // bias inherent
    var fmm = try n.fastNetworkSolver(allocator);
    defer fmm.deinit();
    var erred = false;
    fmm.loadSensors(&d1) catch {
        erred = true;
    };
    try std.testing.expect(!erred);

    var fmm_impl: *FastModularNetworkSolver = @ptrCast(@alignCast(fmm.ptr));
    // test that network has active signals
    var active = countActiveSignals(fmm_impl);
    try std.testing.expect(active != 0);

    // flush and test
    var res = try fmm.flush();
    try std.testing.expect(res);

    active = countActiveSignals(fmm_impl);
    try std.testing.expect(active == 0);
}

test "FastModularNetworkSolver node count" {
    const allocator = std.testing.allocator;
    var n = try buildModularNetwork(allocator);
    defer n.deinit();

    var fmm = try n.fastNetworkSolver(allocator);
    defer fmm.deinit();
    try std.testing.expect(fmm.nodeCount() == 9);
}

test "FastModularNetworkSolver link count" {
    const allocator = std.testing.allocator;
    var n = try buildModularNetwork(allocator);
    defer n.deinit();

    var fmm = try n.fastNetworkSolver(allocator);
    defer fmm.deinit();
    try std.testing.expect(fmm.linkCount() == 9);
}
