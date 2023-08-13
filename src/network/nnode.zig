const std = @import("std");
const net_common = @import("common.zig");
const net_link = @import("link.zig");
const neat_math = @import("../math/activations.zig");
const neat_trait = @import("../trait.zig");

const testing = std.testing;

const traitWithId = @import("../genetics/common.zig").traitWithId;
const NodeNeuronType = net_common.NodeNeuronType;
const NodeActivationType = neat_math.NodeActivationType;
const NodeType = net_common.NodeType;
const Link = net_link.Link;
const Trait = neat_trait.Trait;

pub const NNodeJSON = struct {
    id: i64,
    trait_id: ?i64,
    neuron_type: NodeNeuronType,
    activation: neat_math.NodeActivationType,
};

/// NNode is either a NEURON or a SENSOR. If it's a sensor, it can be loaded with a value for output.
/// If it's a neuron, it has a list of its incoming input signals ([]*Link is used).
/// Use an activation count to avoid flushing.
pub const NNode = struct {
    /// The node id.
    id: i64 = 0,
    /// The type of node activation function.
    activation_type: neat_math.NodeActivationType,
    /// The neuron type for this node.
    neuron_type: NodeNeuronType,
    /// The node's activation value.
    activation: f64 = 0,
    /// The number of activations for this node.
    activations_count: i32 = 0,
    /// The activation sum.
    activation_sum: f64 = 0,

    /// List of all incoming connections.
    incoming: std.ArrayList(*Link),
    /// List of all outgoing connections.
    outgoing: std.ArrayList(*Link),
    /// The trait linked to node.
    trait: ?*Trait = null,
    /// Used for Gene decoding by referencing analogue to this node in Organism phenotype.
    phenotype_analogue: *NNode = undefined,
    /// The flag used for loop detection.
    visited: bool = false,

    /// Learning Parameters; the following parameters are for use in neurons that
    /// learn through habituation, sensitization, or Hebbian-type processes.
    params: []f64 = undefined,
    /// Denotes whether params is allocated (used internally when calling deinit).
    has_params: bool = false,

    /// The activation value at time t-1; holds the previous step's
    /// activation value for recurrency.
    last_activation: f64 = 0,
    /// Activation value of node at time t-2; holds the activation before the previous step's.
    /// This is necessary for a special recurrent case when the in_node of a recurrent link is
    /// one time step ahead of the outnode. The in_node then needs to send from TWO time steps ago.
    last_activation_2: f64 = 0,
    /// If true, node is active; used during node activation.
    is_active: bool = false,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// For internal use only; parses NNode into format that can be serialized to JSON.
    pub fn jsonify(self: *NNode) NNodeJSON {
        return .{
            .id = self.id,
            .trait_id = if (self.trait != null) self.trait.?.id.? else null,
            .neuron_type = self.neuron_type,
            .activation = self.activation_type,
        };
    }

    /// Initializes a new NNode with default values.
    pub fn rawInit(allocator: std.mem.Allocator) !*NNode {
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

    /// Initializes a new NNode from JSON (used when reading Genome from file).
    pub fn initFromJSON(allocator: std.mem.Allocator, value: NNodeJSON, traits: []*Trait) !*NNode {
        var node = try NNode.rawInit(allocator);
        node.id = value.id;
        node.neuron_type = value.neuron_type;
        node.activation_type = value.activation;
        var trait = traitWithId(if (value.trait_id == null) 0 else value.trait_id.?, traits);
        if (trait != null) node.trait = trait.?;
        return node;
    }

    /// Initializes a new NNode with specified Id and neuron type.
    pub fn init(allocator: std.mem.Allocator, node_id: i64, neuron_type: NodeNeuronType) !*NNode {
        var node = try NNode.rawInit(allocator);
        node.id = node_id;
        node.neuron_type = neuron_type;
        return node;
    }

    /// Initializes a new NNode from an existing NNode with given trait for Genome purposes.
    pub fn initCopy(allocator: std.mem.Allocator, n: *NNode, t: ?*Trait) !*NNode {
        var node = try NNode.rawInit(allocator);
        node.id = n.id;
        node.neuron_type = n.neuron_type;
        node.activation_type = n.activation_type;
        node.trait = t;
        return node;
    }

    /// Initializes a new NNode from file (used when reading Genome from plain text file).
    pub fn readFromFile(allocator: std.mem.Allocator, data: []const u8, traits: []*Trait) !*NNode {
        var node = try NNode.rawInit(allocator);
        errdefer node.deinit();
        var split = std.mem.split(u8, data, " ");
        // parse node id
        var count: usize = 0;
        while (split.next()) |d| : (count += 1) {
            if (count == 2) continue;
            switch (count) {
                0 => node.id = try std.fmt.parseInt(i64, d, 10),
                1 => {
                    var trait_id = try std.fmt.parseInt(i64, d, 10);
                    node.trait = traitWithId(trait_id, traits);
                },
                3 => {
                    var neuron_type_u8 = try std.fmt.parseInt(u8, d, 10);
                    node.neuron_type = @as(NodeNeuronType, @enumFromInt(neuron_type_u8));
                },
                4 => {
                    node.activation_type = neat_math.NodeActivationType.activationTypeByName(d);
                },
                else => continue,
            }
        }
        if (count < 3) return error.MalformedNodeInGenomeFile;
        return node;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *NNode) void {
        self.incoming.deinit();
        self.outgoing.deinit();
        self.allocator.destroy(self);
    }

    /// Set new activation value for this NNode.
    pub fn setActivation(self: *NNode, input: f64) void {
        self.saveActivations();
        self.activation = input;
        self.activations_count += 1;
    }

    /// Saves NNode's current activations for potential time delayed connections.
    pub fn saveActivations(self: *NNode) void {
        self.last_activation_2 = self.last_activation;
        self.last_activation = self.activation;
    }

    /// Returns activation for a current step.
    pub fn getActiveOut(self: *NNode) f64 {
        if (self.activations_count > 0) {
            return self.activation;
        } else {
            return 0.0;
        }
    }

    /// Returns activation from PREVIOUS time step.
    pub fn getActiveOutTd(self: *NNode) f64 {
        if (self.activations_count > 1) {
            return self.last_activation;
        } else {
            return 0.0;
        }
    }

    /// Tests equality of NNode against another NNode.
    pub fn isEql(self: *NNode, n: *NNode) bool {
        // check for equality of primitive types
        if (self.id != n.id or self.activation_type != n.activation_type or self.neuron_type != n.neuron_type or self.activation != n.activation or self.activation_sum != n.activation_sum or self.activations_count != n.activations_count or self.visited != n.visited or self.has_params != n.has_params or self.last_activation != n.last_activation or self.last_activation_2 != n.last_activation_2 or self.is_active != n.is_active) {
            return false;
        }

        // validate trait equality
        if ((self.trait != null and n.trait == null) or (self.trait == null and n.trait != null)) {
            return false;
        } else if (self.trait != null and n.trait != null) {
            if (!self.trait.?.isEql(n.trait.?)) {
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
            if (!l.isEql(n.incoming.items[i])) {
                return false;
            }
        }
        // check outgoing links
        if (self.outgoing.items.len != n.outgoing.items.len) {
            return false;
        }
        for (self.outgoing.items, 0..) |l, i| {
            // TODO: validate more than just the link id???
            if (!l.isEql(n.outgoing.items[i])) {
                return false;
            }
        }

        return true;
    }

    /// Returns true if this node is SENSOR.
    pub fn isSensor(self: *const NNode) bool {
        return self.neuron_type == NodeNeuronType.InputNeuron or self.neuron_type == NodeNeuronType.BiasNeuron;
    }

    /// Returns true if this node is NEURON.
    pub fn isNeuron(self: *NNode) bool {
        return self.neuron_type == NodeNeuronType.HiddenNeuron or self.neuron_type == NodeNeuronType.OutputNeuron;
    }

    /// If the node is a SENSOR, returns TRUE and loads the value.
    pub fn sensorLoad(self: *NNode, load: f64) bool {
        if (self.isSensor()) {
            self.saveActivations();
            self.activations_count += 1;
            self.activation = load;
            return true;
        } else {
            return false;
        }
    }

    /// Adds a non-recurrent outgoing link to this NNode. Should be used with caution because this doesn't
    /// create full duplex link needed for proper network activation. This method is only intended for
    /// linking the control nodes. For all other needs use `connectFrom` which properly creates all needed links.
    pub fn addOutgoing(self: *NNode, allocator: std.mem.Allocator, out: *NNode, weight: f64) !*Link {
        var new_link = try Link.init(allocator, weight, self, out, false);
        try self.outgoing.append(new_link);
        return new_link;
    }

    /// Adds a non-recurrent incoming link to this NNode. Should be used with caution because this doesn't
    /// create full duplex link needed for proper network activation. This method only intended for
    /// linking the control nodes. For all other needs use `connectFrom` which properly creates all needed links.
    pub fn addIncoming(self: *NNode, allocator: std.mem.Allocator, in: *NNode, weight: f64) !*Link {
        var new_link = try Link.init(allocator, weight, in, self, false);
        try self.incoming.append(new_link);
        return new_link;
    }

    /// Used to create a Link between two NNodes. The incoming links of current NNode and outgoing
    /// links of `in` NNode would be updated to have reference to the new Link.
    pub fn connectFrom(self: *NNode, allocator: std.mem.Allocator, in: *NNode, weight: f64) !*Link {
        var new_link = try Link.init(allocator, weight, in, self, false);
        try self.incoming.append(new_link);
        try in.outgoing.append(new_link);
        return new_link;
    }

    /// Recursively deactivate backwards through the Network.
    pub fn flushback(self: *NNode) void {
        self.activations_count = 0;
        self.activation = 0;
        self.last_activation = 0;
        self.last_activation_2 = 0;
        self.is_active = false;
        self.visited = false;
    }

    /// Used to verify flushing for debugging.
    pub fn flushbackCheck(self: *NNode) !void {
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

    /// Find the greatest depth starting from this neuron at depth d. If max_depth > 0 it
    /// can be used to stop early in case if very deep network detected.
    pub fn depth(self: *NNode, d: i64, max_depth: i64) !i64 {
        if (max_depth > 0 and d > max_depth) {
            return error.MaximalNetDepthExceeded;
        }
        self.visited = true;
        if (self.isSensor()) {
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

    /// Convenient method to check Network's node type (SENSOR, NEURON).
    pub fn nodeType(self: *const NNode) NodeType {
        if (self.isSensor()) {
            return NodeType.SensorNode;
        }
        return NodeType.NeuronNode;
    }

    /// Formats NNode for printing to writer.
    pub fn format(value: NNode, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        const activation = neat_math.NodeActivationType.activationNameByType(value.activation_type);
        var active: []const u8 = "active";
        if (!value.is_active) {
            active = "inactive";
        }

        var used_params: []f64 = &[0]f64{};
        if (value.has_params) {
            used_params = value.params;
        }
        return writer.print("({s} id:{d}, {s}, {s},\t{s} -> step: {d} = {d:.3} {any})", .{ net_common.nodeTypeName(value.nodeType()), value.id, net_common.neuronTypeName(value.neuron_type), activation, active, value.activations_count, value.activation, used_params });
    }
};

test "NNode `init`" {
    const allocator = testing.allocator;
    var node = try NNode.rawInit(allocator);
    defer node.deinit();
    try testing.expectEqual(node.activation_type, NodeActivationType.SigmoidSteepenedActivation);
    try testing.expectEqual(node.neuron_type, NodeNeuronType.HiddenNeuron);
}

test "NNode `new_NNode`" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();

    try testing.expectEqual(node.id, 1);
    try testing.expectEqual(node.activation_type, NodeActivationType.SigmoidSteepenedActivation);
    try testing.expectEqual(node.neuron_type, NodeNeuronType.InputNeuron);
}

test "NNode `new_NNode_copy`" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
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

    var node_copy = try NNode.initCopy(allocator, node, trait);
    defer node_copy.deinit();

    try testing.expectEqual(node.id, node_copy.id);
    try testing.expectEqual(node.activation_type, node_copy.activation_type);
    try testing.expectEqual(node.neuron_type, node_copy.neuron_type);
    try testing.expectEqual(trait, node_copy.trait.?);
}

test "NNode `sensorLoad`" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();

    var load: f64 = 21.0;
    var res = node.sensorLoad(load);
    try testing.expect(res);
    try testing.expectEqual(node.activations_count, 1);
    try testing.expectEqual(load, node.activation);
    try testing.expectEqual(load, node.getActiveOut());

    var load_2: f64 = 36.0;
    res = node.sensorLoad(load_2);
    try testing.expect(res);
    try testing.expectEqual(node.activations_count, 2);
    try testing.expectEqual(load_2, node.activation);
    // validate activation & time delayed activation
    try testing.expectEqual(load_2, node.getActiveOut());
    try testing.expectEqual(load, node.getActiveOutTd());

    // validate attempting to load incorrect node type returns false
    var hidden_node = try NNode.init(allocator, 1, NodeNeuronType.HiddenNeuron);
    defer hidden_node.deinit();
    res = hidden_node.sensorLoad(load);
    try testing.expect(!res);
}

test "NNode `addIncoming`" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.init(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var weight: f64 = 1.5;

    _ = try node_2.addIncoming(allocator, node, weight);

    try testing.expectEqual(node_2.incoming.items.len, 1);
    try testing.expectEqual(node.outgoing.items.len, 0);

    var link = node_2.incoming.items[0];
    defer link.deinit();

    try testing.expectEqual(weight, link.cxn_weight);
    try testing.expectEqual(node, link.in_node.?);
    try testing.expectEqual(node_2, link.out_node.?);
}

test "NNode `addOutgoing`" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.init(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var weight: f64 = 1.5;

    _ = try node.addOutgoing(allocator, node_2, weight);

    try testing.expectEqual(node.outgoing.items.len, 1);
    try testing.expectEqual(node_2.incoming.items.len, 0);

    var link = node.outgoing.items[0];
    defer link.deinit();

    try testing.expectEqual(weight, link.cxn_weight);
    try testing.expectEqual(node, link.in_node.?);
    try testing.expectEqual(node_2, link.out_node.?);
}

test "NNode `connectFrom`" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.init(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();

    var weight: f64 = 1.5;

    _ = try node_2.connectFrom(allocator, node, weight);

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
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.init(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var node_3 = try NNode.init(allocator, 3, NodeNeuronType.OutputNeuron);
    defer node_3.deinit();

    var link_1 = try node_2.addIncoming(allocator, node, 15.0);
    defer link_1.deinit();
    var link_2 = try node_3.addIncoming(allocator, node_2, 20.0);
    defer link_2.deinit();

    var depth = try node_3.depth(0, 0);
    try testing.expectEqual(depth, 2);
}

test "NNode `depth` with loop" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.init(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var node_3 = try NNode.init(allocator, 3, NodeNeuronType.OutputNeuron);
    defer node_3.deinit();

    var link_1 = try node_2.addIncoming(allocator, node, 15.0);
    defer link_1.deinit();
    var link_2 = try node_3.addIncoming(allocator, node_2, 20.0);
    defer link_2.deinit();
    var link_3 = try node_3.addIncoming(allocator, node_3, 10.0);
    defer link_3.deinit();

    var depth = try node_3.depth(0, 0);
    try testing.expectEqual(depth, 2);
}

test "NNode `depth` with max depth" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    var node_2 = try NNode.init(allocator, 2, NodeNeuronType.HiddenNeuron);
    defer node_2.deinit();
    var node_3 = try NNode.init(allocator, 3, NodeNeuronType.OutputNeuron);
    defer node_3.deinit();

    var link_1 = try node_2.addIncoming(allocator, node, 15.0);
    defer link_1.deinit();
    var link_2 = try node_3.addIncoming(allocator, node_2, 20.0);
    defer link_2.deinit();

    var max_depth: i64 = 1;
    var depth = node_3.depth(0, max_depth);
    try testing.expectError(error.MaximalNetDepthExceeded, depth);
}

test "NNode `flushback`" {
    const allocator = testing.allocator;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();

    var load: f64 = 34.0;
    var load_2: f64 = 14.0;

    _ = node.sensorLoad(load);
    _ = node.sensorLoad(load_2);

    // validate node state is updated
    try testing.expectEqual(node.activations_count, 2);
    try testing.expectEqual(node.activation, 14.0);

    // validate activation and time delayed activation
    try testing.expectEqual(load_2, node.getActiveOut());
    try testing.expectEqual(load, node.getActiveOutTd());

    node.flushback();

    // validate flushback resets node state
    try testing.expectEqual(node.activations_count, 0);
    try testing.expectEqual(node.activation, 0.0);

    // validate activation and time delayed activation
    try testing.expectEqual(node.getActiveOut(), 0.0);
    try testing.expectEqual(node.getActiveOutTd(), 0.0);
}

test "NNode `getActiveOut`" {
    const allocator = testing.allocator;
    var activation: f64 = 1293.98;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    node.activation = activation;

    var out = node.getActiveOut();
    try testing.expectEqual(out, 0.0);

    node.activations_count = 1;
    out = node.getActiveOut();
    try testing.expectEqual(out, activation);
}

test "NNode `getActiveOutTd`" {
    const allocator = testing.allocator;
    var activation: f64 = 1293.98;
    var node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node.deinit();
    node.activation = activation;
    node.activations_count = 1;

    var out = node.getActiveOutTd();
    try testing.expectEqual(out, 0.0);

    node.activations_count = 2;
    out = node.getActiveOut();
    try testing.expectEqual(out, activation);
}

test "NNode `isSensor`" {
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
        var node = try NNode.init(allocator, 1, case.n_type);
        defer node.deinit();
        try testing.expectEqual(case.is_sensor, node.isSensor());
    }
}

test "NNode `isNeuron`" {
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
        var node = try NNode.init(allocator, 1, case.n_type);
        defer node.deinit();
        try testing.expectEqual(case.is_neuron, node.isNeuron());
    }
}

// TODO: test "NNode `flushbackCheck`" {}

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
        var node = try NNode.init(allocator, 1, case.neuron_type);
        defer node.deinit();
        try testing.expectEqual(case.node_type, node.nodeType());
    }
}

// TODO: test "NNode" print debug {}

// TODO: test "NNode string" {}
