const std = @import("std");
const neat_math = @import("../math/activations.zig");
const fast_net = @import("fast_network.zig");

// exports
pub const Link = @import("link.zig").Link;
pub const Network = @import("network.zig").Network;
pub const NNode = @import("nnode.zig").NNode;
pub const Solver = @import("solver.zig").Solver;
pub const FastNetworkLink = fast_net.FastNetworkLink;
pub const FastControlNode = fast_net.FastControlNode;
pub const FastModularNetworkSolver = fast_net.FastModularNetworkSolver;

/// NodeType defines the type of NNode to create.
pub const NodeType = enum(u8) {
    /// Neuron type.
    NeuronNode,
    /// Sensor type.
    SensorNode,
};

/// Returns human-readable NNode type name for given constant value.
pub fn nodeTypeName(node_type: NodeType) []const u8 {
    return switch (node_type) {
        NodeType.NeuronNode => "NEURON",
        NodeType.SensorNode => "SENSOR",
    };
}

/// NodeNeuronType defines the type of neuron to create.
pub const NodeNeuronType = enum(u8) {
    /// HiddenNeuron - the node is in hidden layer.
    HiddenNeuron,
    /// InputNeuron - the node is in input layer.
    InputNeuron,
    /// OutputNeuron - the node is in output layer.
    OutputNeuron,
    /// BiasNeuron - the node is bias
    BiasNeuron,
};

/// Returns human-readable neuron type name for given constant.
pub fn neuronTypeName(neuron_type: NodeNeuronType) []const u8 {
    return switch (neuron_type) {
        NodeNeuronType.HiddenNeuron => "HIDN",
        NodeNeuronType.InputNeuron => "INPT",
        NodeNeuronType.OutputNeuron => "OUTP",
        NodeNeuronType.BiasNeuron => "BIAS",
    };
}

/// Returns neuron node type from its name.
pub fn neuronTypeByName(name: []const u8) NodeNeuronType {
    if (std.mem.eql(u8, name, "HIDN")) {
        return NodeNeuronType.HiddenNeuron;
    }

    if (std.mem.eql(u8, name, "INPT")) {
        return NodeNeuronType.InputNeuron;
    }

    if (std.mem.eql(u8, name, "OUTP")) {
        return NodeNeuronType.OutputNeuron;
    }

    if (std.mem.eql(u8, name, "BIAS")) {
        return NodeNeuronType.BiasNeuron;
    }
    unreachable;
}

/// Used to calculate activation for specified neuron node based on it's activation_type
/// field value. Will return error if unsupported activation type requested.
pub fn activateNode(node: *NNode) !void {
    var res = try neat_math.NodeActivationType.activateByType(node.activation_sum, if (node.has_params) node.params else null, node.activation_type);
    node.setActivation(res);
}

/// Used to activate neuron module presented by provided node. As a result of
/// execution the activation values of all input nodes will be processed by
/// corresponding activation function and corresponding activation values of output
/// nodes will be set. Will panic if unsupported activation type requested.
pub fn activateModule(module: *NNode) !void {
    var inputs = try module.allocator.alloc(f64, module.incoming.items.len);
    defer module.allocator.free(inputs);

    for (module.incoming.items, 0..) |v, i| {
        inputs[i] = v.in_node.?.getActiveOut();
    }

    var outputs = try neat_math.NodeActivationType.activateModuleByType(inputs, if (module.has_params) module.params else null, module.activation_type);
    if (outputs.len != module.outgoing.items.len) {
        std.debug.print("number of output parameters {d} returned by module activator doesn't match the number of output neurons of the module {d}", .{ outputs.len, module.outgoing.items.len });
        return error.ModuleOutputLenMismatch;
    }

    // set outputs
    for (outputs, 0..) |out, i| {
        module.outgoing.items[i].out_node.?.setActivation(out);
        module.outgoing.items[i].out_node.?.is_active = true;
    }
}

test {
    std.testing.refAllDecls(@This());
}
