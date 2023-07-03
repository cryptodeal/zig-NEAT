const std = @import("std");
const net_common = @import("common.zig");
const net_link = @import("link.zig");
const neat_math = @import("../math/activations.zig");
const neat_trait = @import("../trait.zig");

const testing = std.testing;

const NodeNeuronType = net_common.NodeNeuronType;
const NodeActivationType = neat_math.NodeActivationType;
const NodeType = net_common.NodeType;
const Link = net_link.Link;
const Trait = neat_trait.Trait;

pub const NNode = struct {
    // id of the node
    id: i64 = 0,
    // type of node activation fn
    activation_type: neat_math.NodeActivationType,
    // node neuron type
    neuron_type: NodeNeuronType,
    // node activation value
    activation: f64 = 0,
    // number of activations for node
    activations_count: i32 = 0,
    // sum of activations
    activation_sum: f64 = 0,

    // list of all incoming cxns
    incoming: std.ArrayList(*Link),
    // list of all outgoing cxns
    outgoing: std.ArrayList(*Link),
    // trait linked to node
    trait: ?*Trait = null,
    // Used for Gene decoding by referencing analogue to this node in organism phenotype
    phenotype_analogue: *NNode = undefined,
    // flag used for loop detection
    visited: bool = false,

    // learning parameters
    params: []f64 = undefined,
    has_params: bool = false,

    // activation value at time t-1
    // holds prev step activation val for recurrency
    last_activation: f64 = 0,
    // activation value at time t-2
    last_activation_2: f64 = 0,
    // if true, node is active; used during node activation
    is_active: bool = false,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*NNode {
        var node = try allocator.create(NNode);
        node.* = .{
            .allocator = allocator,
            .neuron_type = NodeNeuronType.HiddenNeuron,
            .activation_type = NodeActivationType.SigmoidSteepenedActivation,
            .incoming = std.ArrayList(*Link).init(allocator),
            .outgoing = std.ArrayList(*Link).init(allocator),
        };
        return node;
    }

    pub fn new_NNode(allocator: std.mem.Allocator, node_id: i64, neuron_type: NodeNeuronType) !*NNode {
        var node = try NNode.init(allocator);
        node.id = node_id;
        node.neuron_type = neuron_type;
        return node;
    }

    pub fn new_NNode_copy(allocator: std.mem.Allocator, n: *NNode, t: ?*Trait) !*NNode {
        var node = try NNode.init(allocator);
        node.id = n.id;
        node.neuron_type = n.neuron_type;
        node.activation_type = n.activation_type;
        node.trait = t;
        return node;
    }

    pub fn deinit(self: *NNode) void {
        self.incoming.deinit();
        self.outgoing.deinit();
        self.allocator.destroy(self);
    }

    pub fn set_activation(self: *NNode, input: f64) void {
        self.save_activations();
        self.activation = input;
        self.activations_count += 1;
    }

    pub fn save_activations(self: *NNode) void {
        self.last_activation_2 = self.last_activation;
        self.last_activation = self.activation;
    }

    pub fn get_active_out(self: *NNode) f64 {
        if (self.activations_count > 0) {
            return self.activation;
        } else {
            return 0.0;
        }
    }

    pub fn get_active_out_td(self: *NNode) f64 {
        if (self.activations_count > 1) {
            return self.last_activation;
        } else {
            return 0.0;
        }
    }

    pub fn is_equal(self: *NNode, n: *NNode) bool {
        // check for equality of primitive types
        if (self.id != n.id or self.activation_type != n.activation_type or self.neuron_type != n.neuron_type or self.activation != n.activation or self.activation_sum != n.activation_sum or self.activations_count != n.activations_count or self.visited != n.visited or self.has_params != n.has_params or self.last_activation != n.last_activation or self.last_activation_2 != n.last_activation_2 or self.is_active != n.is_active) {
            return false;
        }

        // validate trait equality
        if ((self.trait != null and n.trait == null) or (self.trait == null and n.trait != null)) {
            return false;
        } else if (self.trait != null and n.trait != null) {
            if (!self.trait.?.is_equal(n.trait.?)) {
                return false;
            }
        }
        // validate params
        if (self.has_params != n.has_params) {
            return false;
        } else if (self.has_params and n.has_params and !std.mem.eql(f64, self.params, n.params)) {
            return false;
        }
        // check incoming links
        if (self.incoming.items.len != n.incoming.items.len) {
            return false;
        }
        for (self.incoming.items, 0..) |l, i| {
            if (!l.is_equal(n.incoming.items[i])) {
                return false;
            }
        }
        // check outgoing links
        if (self.outgoing.items.len != n.outgoing.items.len) {
            return false;
        }
        for (self.outgoing.items, 0..) |l, i| {
            // TODO: validate more than just the link id???
            if (!l.is_equal(n.outgoing.items[i])) {
                return false;
            }
        }

        return true;
    }

    pub fn is_sensor(self: *const NNode) bool {
        return self.neuron_type == NodeNeuronType.InputNeuron or self.neuron_type == NodeNeuronType.BiasNeuron;
    }

    pub fn is_neuron(self: *NNode) bool {
        return self.neuron_type == NodeNeuronType.HiddenNeuron or self.neuron_type == NodeNeuronType.OutputNeuron;
    }

    pub fn sensor_load(self: *NNode, load: f64) bool {
        if (self.is_sensor()) {
            self.save_activations();
            self.activations_count += 1;
            self.activation = load;
            return true;
        } else {
            return false;
        }
    }

    pub fn add_outgoing(self: *NNode, out: *NNode, weight: f64) !*Link {
        var new_link = try Link.init(self.allocator, weight, self, out, false);
        try self.outgoing.append(new_link);
        return new_link;
    }

    pub fn add_incoming(self: *NNode, in: *NNode, weight: f64) !*Link {
        var new_link = try Link.init(self.allocator, weight, in, self, false);
        try self.incoming.append(new_link);
        return new_link;
    }

    pub fn connect_from(self: *NNode, in: *NNode, weight: f64) !*Link {
        var new_link = try Link.init(self.allocator, weight, in, self, false);
        try self.incoming.append(new_link);
        try in.outgoing.append(new_link);
        return new_link;
    }

    pub fn flushback(self: *NNode) void {
        self.activations_count = 0;
        self.activation = 0;
        self.last_activation = 0;
        self.last_activation_2 = 0;
        self.is_active = false;
        self.visited = false;
    }

    pub fn flushback_check(self: *NNode) !void {
        if (self.activations_count > 0) {
            std.debug.print("NNODE: {s} has activation count {d}", .{ self, self.activations_count });
            return error.NNodeFlushbackError;
        }
        if (self.activation > 0) {
            std.debug.print("NNODE: {s} has activation {d}", .{ self, self.activation });
            return error.NNodeFlushbackError;
        }
        if (self.last_activation > 0) {
            std.debug.print("NNODE: {s} has last_activation {d}", .{ self, self.last_activation });
            return error.NNodeFlushbackError;
        }
        if (self.last_activation_2 > 0) {
            std.debug.print("NNODE: {s} has last_activation_2 {d}", .{ self, self.last_activation_2 });
            return error.NNodeFlushbackError;
        }
    }

    pub fn depth(self: *NNode, d: i64, max_depth: i64) !i64 {
        if (max_depth > 0 and d > max_depth) {
            return error.MaximalNetDepthExceeded;
        }
        self.visited = true;
        if (self.is_sensor()) {
            return d;
        } else {
            var max: i64 = d;
            for (self.incoming.items) |l| {
                if (l.in_node.?.visited) {
                    continue;
                }
                var curr_depth = try l.in_node.?.depth(d + 1, max_depth);
                if (curr_depth > max) {
                    max = curr_depth;
                }
            }
            return max;
        }
    }

    pub fn node_type(self: *const NNode) NodeType {
        if (self.is_sensor()) {
            return NodeType.SensorNode;
        }
        return NodeType.NeuronNode;
    }

    pub fn format(value: NNode, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        const activation = neat_math.NodeActivationType.activation_name_by_type(value.activation_type);
        var active: []const u8 = "active";
        if (!value.is_active) {
            active = "inactive";
        }

        var used_params: []f64 = &[0]f64{};
        if (value.has_params) {
            used_params = value.params;
        }
        return writer.print("({s} id:{d}, {s}, {s},\t{s} -> step: {d} = {d:.3} {any})", .{ net_common.node_type_name(value.node_type()), value.id, net_common.neuron_type_name(value.neuron_type), activation, active, value.activations_count, value.activation, used_params });
    }
};

test "NNode `init`" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator);
    defer node.deinit();
    try testing.expectEqual(node.activation_type, NodeActivationType.SigmoidSteepenedActivation);
    try testing.expectEqual(node.neuron_type, NodeNeuronType.HiddenNeuron);
}

test "NNode `new_NNode`" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();

    try testing.expectEqual(node.id, 1);
    try testing.expectEqual(node.activation_type, NodeActivationType.SigmoidSteepenedActivation);
    try testing.expectEqual(node.neuron_type, NodeNeuronType.InputNeuron);
}

test "NNode `new_NNode_copy`" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();

    var trait: *Trait = try Trait.init(allocator, 6);
    defer trait.deinit();
    // manually set trait weights for test
    trait.params[0] = 1.1;
    trait.params[1] = 2.3;
    trait.params[2] = 3.4;
    trait.params[3] = 4.2;
    trait.params[4] = 5.5;
    trait.params[5] = 6.7;

    var node_copy = try NNode.new_NNode_copy(allocator, node, trait);
    defer node_copy.deinit();

    try testing.expectEqual(node.id, node_copy.id);
    try testing.expectEqual(node.activation_type, node_copy.activation_type);
    try testing.expectEqual(node.neuron_type, node_copy.neuron_type);
    try testing.expectEqual(trait, node_copy.trait.?);
}

test "NNode `sensor_load`" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();

    var load: f64 = 21.0;
    var res = node.sensor_load(load);
    try testing.expect(res);
    try testing.expectEqual(node.activations_count, 1);
    try testing.expectEqual(load, node.activation);
    try testing.expectEqual(load, node.get_active_out());

    var load_2: f64 = 36.0;
    res = node.sensor_load(load_2);
    try testing.expect(res);
    try testing.expectEqual(node.activations_count, 2);
    try testing.expectEqual(load_2, node.activation);
    // validate activation & time delayed activation
    try testing.expectEqual(load_2, node.get_active_out());
    try testing.expectEqual(load, node.get_active_out_td());

    // validate attempting to load incorrect node type returns false
    var hidden_node = try NNode.new_NNode(allocator, 1, NodeNeuronType.HiddenNeuron);
    defer hidden_node.deinit();
    res = hidden_node.sensor_load(load);
    try testing.expect(!res);
}

test "NNode `add_incoming`" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.new_NNode(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var weight: f64 = 1.5;

    _ = try node_2.add_incoming(node, weight);

    try testing.expectEqual(node_2.incoming.items.len, 1);
    try testing.expectEqual(node.outgoing.items.len, 0);

    var link = node_2.incoming.items[0];
    defer link.deinit();

    try testing.expectEqual(weight, link.cxn_weight);
    try testing.expectEqual(node, link.in_node.?);
    try testing.expectEqual(node_2, link.out_node.?);
}

test "NNode `add_outgoing`" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.new_NNode(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var weight: f64 = 1.5;

    _ = try node.add_outgoing(node_2, weight);

    try testing.expectEqual(node.outgoing.items.len, 1);
    try testing.expectEqual(node_2.incoming.items.len, 0);

    var link = node.outgoing.items[0];
    defer link.deinit();

    try testing.expectEqual(weight, link.cxn_weight);
    try testing.expectEqual(node, link.in_node.?);
    try testing.expectEqual(node_2, link.out_node.?);
}

test "NNode `connect_from`" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.new_NNode(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();

    var weight: f64 = 1.5;

    _ = try node_2.connect_from(node, weight);

    try testing.expectEqual(node_2.incoming.items.len, 1);
    try testing.expectEqual(node.outgoing.items.len, 1);

    var link = node_2.incoming.items[0];
    defer link.deinit();

    try testing.expectEqual(weight, link.cxn_weight);
    try testing.expectEqual(node, link.in_node.?);
    try testing.expectEqual(node_2, link.out_node.?);
    try testing.expectEqual(link, node.outgoing.items[0]);
}

test "NNode `depth`" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.new_NNode(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var node_3 = try NNode.new_NNode(allocator, 3, NodeNeuronType.OutputNeuron);
    defer node_3.deinit();

    var link_1 = try node_2.add_incoming(node, 15.0);
    defer link_1.deinit();
    var link_2 = try node_3.add_incoming(node_2, 20.0);
    defer link_2.deinit();

    var depth = try node_3.depth(0, 0);
    try testing.expectEqual(depth, 2);
}

test "NNode `depth` with loop" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.new_NNode(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var node_3 = try NNode.new_NNode(allocator, 3, NodeNeuronType.OutputNeuron);
    defer node_3.deinit();

    var link_1 = try node_2.add_incoming(node, 15.0);
    defer link_1.deinit();
    var link_2 = try node_3.add_incoming(node_2, 20.0);
    defer link_2.deinit();
    var link_3 = try node_3.add_incoming(node_3, 10.0);
    defer link_3.deinit();

    var depth = try node_3.depth(0, 0);
    try testing.expectEqual(depth, 2);
}

test "NNode `depth` with max depth" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.new_NNode(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var node_3 = try NNode.new_NNode(allocator, 3, NodeNeuronType.OutputNeuron);
    defer node_3.deinit();

    var link_1 = try node_2.add_incoming(node, 15.0);
    defer link_1.deinit();
    var link_2 = try node_3.add_incoming(node_2, 20.0);
    defer link_2.deinit();

    var max_depth: i64 = 1;
    var depth = node_3.depth(0, max_depth);
    try testing.expectError(error.MaximalNetDepthExceeded, depth);
}

test "NNode `flushback`" {
    const allocator = testing.allocator;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();

    var load: f64 = 34.0;
    var load_2: f64 = 14.0;

    _ = node.sensor_load(load);
    _ = node.sensor_load(load_2);

    // validate node state is updated
    try testing.expectEqual(node.activations_count, 2);
    try testing.expectEqual(node.activation, 14.0);

    // validate activation and time delayed activation
    try testing.expectEqual(load_2, node.get_active_out());
    try testing.expectEqual(load, node.get_active_out_td());

    node.flushback();

    // validate flushback resets node state
    try testing.expectEqual(node.activations_count, 0);
    try testing.expectEqual(node.activation, 0.0);

    // validate activation and time delayed activation
    try testing.expectEqual(node.get_active_out(), 0.0);
    try testing.expectEqual(node.get_active_out_td(), 0.0);
}

test "NNode `get_active_out`" {
    const allocator = testing.allocator;
    var activation: f64 = 1293.98;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    node.activation = activation;

    var out = node.get_active_out();
    try testing.expectEqual(out, 0.0);

    node.activations_count = 1;
    out = node.get_active_out();
    try testing.expectEqual(out, activation);
}

test "NNode `get_active_out_td`" {
    const allocator = testing.allocator;
    var activation: f64 = 1293.98;
    var node = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    node.activation = activation;
    node.activations_count = 1;

    var out = node.get_active_out_td();
    try testing.expectEqual(out, 0.0);

    node.activations_count = 2;
    out = node.get_active_out();
    try testing.expectEqual(out, activation);
}

test "NNode `is_sensor`" {
    const allocator = testing.allocator;
    var test_cases = [_]struct { n_type: NodeNeuronType, is_sensor: bool }{
        .{
            .n_type = NodeNeuronType.InputNeuron,
            .is_sensor = true,
        },
        .{
            .n_type = NodeNeuronType.BiasNeuron,
            .is_sensor = true,
        },
        .{
            .n_type = NodeNeuronType.HiddenNeuron,
            .is_sensor = false,
        },
        .{
            .n_type = NodeNeuronType.OutputNeuron,
            .is_sensor = false,
        },
    };

    for (test_cases) |case| {
        var node = try NNode.new_NNode(allocator, 1, case.n_type);
        defer node.deinit();
        try testing.expectEqual(case.is_sensor, node.is_sensor());
    }
}

test "NNode `is_neuron`" {
    const allocator = testing.allocator;
    var test_cases = [_]struct { n_type: NodeNeuronType, is_neuron: bool }{
        .{
            .n_type = NodeNeuronType.InputNeuron,
            .is_neuron = false,
        },
        .{
            .n_type = NodeNeuronType.BiasNeuron,
            .is_neuron = false,
        },
        .{
            .n_type = NodeNeuronType.HiddenNeuron,
            .is_neuron = true,
        },
        .{
            .n_type = NodeNeuronType.OutputNeuron,
            .is_neuron = true,
        },
    };

    for (test_cases) |case| {
        var node = try NNode.new_NNode(allocator, 1, case.n_type);
        defer node.deinit();
        try testing.expectEqual(case.is_neuron, node.is_neuron());
    }
}

// TODO: test "NNode `flushback_check`" {}

test "NNode node type" {
    const allocator = testing.allocator;
    var test_cases = [_]struct { neuron_type: NodeNeuronType, node_type: NodeType }{
        .{
            .neuron_type = NodeNeuronType.InputNeuron,
            .node_type = NodeType.SensorNode,
        },
        .{
            .neuron_type = NodeNeuronType.BiasNeuron,
            .node_type = NodeType.SensorNode,
        },
        .{
            .neuron_type = NodeNeuronType.HiddenNeuron,
            .node_type = NodeType.NeuronNode,
        },
        .{
            .neuron_type = NodeNeuronType.OutputNeuron,
            .node_type = NodeType.NeuronNode,
        },
    };

    for (test_cases) |case| {
        var node = try NNode.new_NNode(allocator, 1, case.neuron_type);
        defer node.deinit();
        try testing.expectEqual(case.node_type, node.node_type());
    }
}

// TODO: test "NNode" print debug {}

// TODO: test "NNode string" {}
