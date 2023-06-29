const std = @import("std");
const net_math = @import("../math/activations.zig");
const TransientAllocator = @import("../utils.zig").TransientAllocator;

/// FastNetworkLink The connection descriptor for fast network
pub const FastNetworkLink = struct {
    // index of source neuron
    source_idx: i64,
    // index of target neuron
    target_idx: i64,
    // weight of link
    weight: f64 = 0,
    // signal relayed by link
    signal: f64 = 0,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, source: i64, target: i64, weight: f64) !*FastNetworkLink {
        var self = try allocator.create(FastNetworkLink);
        self.* = .{
            .allocator = allocator,
            .source_idx = source,
            .target_idx = target,
            .weight = weight,
        };
        return self;
    }

    pub fn deinit(self: *FastNetworkLink) void {
        self.allocator.destroy(self);
    }
};

/// FastControlNode The module relay (control node) descriptor for fast network
pub const FastControlNode = struct {
    // activation fn for control node
    activation_type: net_math.NodeActivationType,
    // indexes of input nodes
    input_idxs: []i64,
    // indexes of output nodes
    output_idxs: []i64,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, inputs: []i64, outputs: []i64, activation: net_math.NodeActivationType) !*FastNetworkLink {
        var self = try allocator.create(FastControlNode);
        self.* = .{
            .allocator = allocator,
            .input_idxs = inputs,
            .output_idxs = outputs,
            .activation_type = activation,
        };
        return self;
    }

    pub fn deinit(self: *FastControlNode) void {
        self.allocator.free(self.input_idxs);
        self.allocator.free(self.output_idxs);
        self.allocator.destroy(self);
    }
};

/// FastModularNetworkSolver is the network solver implementation to be used for large neural networks simulation.
pub const FastModularNetworkSolver = struct {
    // allocator for network mem
    mem: TransientAllocator,
    allocator: std.mem.Allocator,

    // network id
    id: i64,
    // network name
    name: []const u8,
    // current activation values per neuron
    neuron_signals: []f64,
    // slice parallels `neuron_signals`; used to test network relaxation
    neuron_signals_processing: []f64,

    // activation functions per neuron, must be ordered same as `neuron_signals`
    // has undefined entries for neurons that are inputs/outputs of module
    activation_functions: []net_math.NodeActivationType,
    // current bias values per neuron
    bias_list: []f64,
    // control nodes relaying between network modules
    modules: []*FastControlNode,
    // connections
    cxns: []*FastNetworkLink,

    // count of input neurons
    input_neuron_count: i64,
    // total count of sensors in the network (input + bias); also index of first output neuron in neuron signals
    sensor_neuron_count: i64,
    // count of output neurons
    output_neuron_count: i64,
    // count of bias neurons (usually 1); also index of first input neuron in neuron signals
    bias_neuron_count: i64,
    // total count of neurons in network
    total_neuron_count: i64,

    // for recursive activation, tracks whether node has been activated
    activated: []bool,
    // for recursive activation, tracks whether node is currently being calculated (recurrent cxns processing)
    in_activation: []bool,
    // for recursive activation, tracks prev activation values of recurrent cxns (recurrent cxns processing)
    last_activation: []f64,

    // holds IDs of outgoing nodes for each network node
    adjacent_list: []std.ArrayList(i64),
    // holds IDs of incoming nodes for each network node
    reverse_adjacent_list: []std.ArrayList(i64),
    // holds cxn weights from all connected nodes
    adjacent_matrix: [][]f64,

    /// initialize new FastModularNetworkSolver
    pub fn init(allocator: std.mem.Allocator, bias_neuron_count: i64, input_neuron_count: i64, output_neuron_count: i64, total_neuron_count: i64, activation_fns: []net_math.NodeActivationType, cxns: []*FastNetworkLink, bias_list: []f64, modules: []*FastControlNode) !*FastModularNetworkSolver {
        var self = try allocator.create(FastModularNetworkSolver);
        self.* = .{
            .allocator = allocator,
            .mem = TransientAllocator.init(allocator),
            .bias_neuron_count = bias_neuron_count,
            .input_neuron_count = input_neuron_count,
            .sensor_neuron_count = bias_neuron_count + input_neuron_count,
            .output_neuron_count = output_neuron_count,
            .total_neuron_count = total_neuron_count,
            .activation_functions = activation_fns,
            .bias_list = bias_list,
            .modules = modules,
            .cxns = cxns,
        };

        // alloc slices that store states throughout network
        // neuron signals are initialized to 0.0
        self.neuron_signals = try self.mem.allocator().alloc(f64, total_neuron_count);
        for (self.neuron_signals) |*x| x.* = 0;
        self.neuron_signals_processing = try self.mem.allocator().alloc(f64, total_neuron_count);
        for (self.neuron_signals_processing) |*x| x.* = 0;

        var i: usize = 0;
        while (i < bias_neuron_count) : (i += 1) {
            // BIAS neuron signal
            self.neuron_signals[i] = 1;
        }

        // alloc slices that store activation state
        self.activated = try self.mem.allocator().alloc(bool, total_neuron_count);
        for (self.activated) |*x| x.* = false;
        self.in_activation = try self.mem.allocator().alloc(bool, total_neuron_count);
        for (self.in_activation) |*x| x.* = false;
        self.last_activation = try self.mem.allocator().alloc(f64, total_neuron_count);
        for (self.last_activation) |*x| x.* = 0;

        // alloc adjacent lists/matrix for fast access to incoming/outgoing nodes & cxn weights
        self.adjacent_list = try self.mem.allocator().alloc(std.ArrayList(i64), total_neuron_count);
        self.reverse_adjacent_list = try self.mem.allocator().alloc(std.ArrayList(i64), total_neuron_count);
        self.adjacent_matrix = try self.mem.allocator().alloc([]f64, total_neuron_count);

        i = 0;
        while (i < total_neuron_count) : (i += 1) {
            self.adjacent_list[i] = std.ArrayList(i64).init(self.mem.allocator());
            self.reverse_adjacent_list[i] = std.ArrayList(i64).init(self.mem.allocator());
            self.adjacent_matrix[i] = try self.mem.allocator().alloc(f64, total_neuron_count);
        }

        i = 0;
        while (i < cxns.len) : (i += 1) {
            var crs = cxns[i].source_idx;
            var crt = cxns[i].target_idx;

            // holds outgoing nodes
            self.adjacent_list[crs].append(crt);
            // holds incoming nodes
            self.reverse_adjacent_list[crt].append(crs);
            // holds link weight
            self.adjacent_matrix[crs][crt] = cxns[i].weight;
        }

        return self;
    }

    /// deinitialize the FastModularNetworkSolver
    pub fn deinit(self: *FastModularNetworkSolver) void {
        self.mem.deinit();
        self.allocator.destroy(self);
    }

    pub fn fwd_steps(self: *FastModularNetworkSolver, steps: i64) !bool {
        var res: bool = undefined;
        var i: usize = 0;
        while (i < steps) : (i += 1) {
            res = try self.fwd_step(0);
        }
        return res;
    }

    pub fn recursive_steps(self: *FastModularNetworkSolver) !bool {
        if (self.modules.len > 0) {
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
            res = try self.recursive_activate_node(index);
            if (!res) {
                std.debug.print("failed to recursively activate the output neuron at {d}", .{index});
                return error.FailedRecursiveStep;
            }
        }
    }

    pub fn recursive_activate_node(self: *FastModularNetworkSolver, current_node: usize) !bool {
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

        while (i < self.reverse_adjacent_list.len) : (i += 1) {
            var current_adj_node: usize = self.reverse_adjacent_list[current_node][i];
            // if node in calculation, reached cycle/recurrent cxn
            // use prev activation value
            if (self.in_activation[current_adj_node]) {
                self.neuron_signals_processing[current_node] += self.last_activation[current_adj_node] * self.adjacent_matrix[current_adj_node][current_node];
            } else {
                // else proceed normally
                // recurse if neuron not yet activated
                if (!self.activated[current_adj_node]) {
                    const res = try self.recursive_activate_node(current_adj_node);
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
        self.neuron_signals[current_node] = try net_math.NodeActivationType.activate_by_type(self.neuron_signals_processing[current_node], undefined, self.activation_functions[current_node]);
        return true;
    }

    pub fn relax(self: *FastModularNetworkSolver, max_steps: usize, max_allowed_signal_delta: f64) !bool {
        var relaxed: bool = false;
        var i: usize = 0;
        while (i < max_steps) : (i += 1) {
            relaxed = try self.forward_step(max_allowed_signal_delta);
            if (relaxed) break;
        }
        return relaxed;
    }

    // TODO: verify that error handling is working as expected
    pub fn fwd_step(self: *FastModularNetworkSolver, max_allowed_signal_delta: f64) !bool {
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
                signal += self.bias_list[i];
            }
            self.neuron_signals_processing[i] = try net_math.NodeActivationType.activate_by_type(signal, undefined, self.activation_functions[i]);
        }

        // pass signals through each module (activation function with more than one input or output)
        for (self.modules) |module| {
            var inputs = try self.allocator.alloc(f64, module.input_idxs.len);
            defer self.allocator.free(inputs);
            for (module.input_idxs, 0..) |inIndex, idx| {
                inputs[idx] = self.neuron_signals_processing[inIndex];
            }
            var outputs = try net_math.NodeActivationType.activate_module_by_type(inputs, undefined, module.activation_type);
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

    pub fn flush(self: *FastModularNetworkSolver) !bool {
        var i: usize = self.bias_neuron_count;
        while (i < self.total_neuron_count) : (i += 1) {
            self.neuron_signals[i] = 0.0;
        }
        return true;
    }

    pub fn load_sensor(self: *FastModularNetworkSolver, inputs: []f64) !void {
        if (inputs.len == self.input_neuron_count) {
            // only inputs should be provided
            var i: usize = 0;
            while (i < self.input_neuron_count) : (i += 1) {
                self.neuron_signals[self.bias_neuron_count + i] = inputs[i];
            }
        } else {
            std.debug.print("the sensors array size is unsupported by network solver", .{});
            return error.ErrNetUnsupportedSensorsArraySize;
        }
    }

    pub fn read_outputs(self: *FastModularNetworkSolver) []f64 {
        // decouple & return
        var outs = try self.mem.allocator().alloc(f64, self.output_neuron_count);
        try @memcpy(outs, self.neuron_signals[self.sensor_neuron_count .. self.sensor_neuron_count + self.output_neuron_count]);
        return outs;
    }

    pub fn node_count(self: *FastModularNetworkSolver) i64 {
        var res: i64 = self.total_neuron_count + self.modules.len;
        return res;
    }

    pub fn link_count(self: *FastModularNetworkSolver) i64 {
        // count all cxns
        var num_links: i64 = self.cxns.len;

        // count all bias links if any
        if (self.bias_neuron_count > 0) {
            for (self.bias_list) |b| {
                if (b != 0) num_links += 1;
            }
        }

        // count all module links
        if (self.modules.len != 0) {
            for (self.modules) |m| {
                num_links += m.input_idxs.len + m.output_idxs.len;
            }
        }
        return num_links;
    }

    pub fn string(self: *FastModularNetworkSolver) ![]const u8 {
        var buffer = std.ArrayList(u8).init(self.mem.allocator());
        buffer.writer().print("FastModularNetwork, id: {d}, name: [{s}], neurons: {d},\n\tinputs: {d},\tbias: {d},\toutputs: {d},\t hidden: {d}", .{ self.id, self.name, self.total_neuron_count, self.input_neuron_count, self.bias_neuron_count, self.output_neuron_count, (self.total_neuron_count - self.sensor_neuron_count - self.output_neuron_count) });
        const res: []const u8 = buffer.toOwnedSlice();
        return res;
    }
};
