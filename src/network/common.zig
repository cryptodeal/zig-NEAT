const std = @import("std");
const NNode = @import("nnode.zig").NNode;
const neat_math = @import("../math/activations.zig");

pub const NodeType = enum(u8) {
    // neuron type
    NeuronNode,
    // sensor type
    SensorNode,
};

pub fn node_type_name(node_type: NodeType) []const u8 {
    return switch (node_type) {
        NodeType.NeuronNode => "NEURON",
        NodeType.SensorNode => "SENSOR",
    };
}

pub const NodeNeuronType = enum(u8) {
    // HiddenNeuron The node is in hidden layer
    HiddenNeuron,
    // InputNeuron The node is in input layer
    InputNeuron,
    // OutputNeuron The node is in output layer
    OutputNeuron,
    // BiasNeuron The node is bias
    BiasNeuron,
};

pub fn neuron_type_name(neuron_type: NodeNeuronType) []const u8 {
    return switch (neuron_type) {
        NodeNeuronType.HiddenNeuron => "HIDN",
        NodeNeuronType.InputNeuron => "INPT",
        NodeNeuronType.OutputNeuron => "OUTP",
        NodeNeuronType.BiasNeuron => "BIAS",
    };
}

pub fn neuron_type_by_name(name: []const u8) NodeNeuronType {
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

    @compileError("Unknown neuron type name: " ++ name);
}

pub fn activate_node(node: *NNode) !void {
    var res = try neat_math.NodeActivationType.activate_by_type(node.activation_sum, node.params, node.activation_type);
    node.set_activation(res);
}

pub fn activate_module(module: *NNode) !void {
    var inputs = try module.allocator.alloc(f64, module.incoming.len);
    defer module.allocator.free(inputs);

    for (module.incoming, 0..) |v, i| {
        inputs[i] = v.in_node.get_active_out();
    }

    var outputs = try neat_math.NodeActivationType.activate_module_by_type(inputs, module.params, module.activation_type);
    if (outputs.len != module.outgoing.len) {
        std.debug.print("number of output parameters {d} returned by module activator doesn't match the number of output neurons of the module {d}", .{ outputs.len, module.outgoing.len });
        return error.ModuleOutputLenMismatch;
    }

    // set outputs
    for (outputs, 0..) |out, i| {
        module.outgoing[i].out_node.set_activation(out);
        module.outgoing[i].out_node.is_active = true;
    }
}
