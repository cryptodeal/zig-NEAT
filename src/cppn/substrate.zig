const std = @import("std");
const neat_activations = @import("../math/activations.zig");
const opts = @import("../opts.zig");
const fast_net = @import("../network/fast_network.zig");
const sub_layout = @import("substrate_layout.zig");
const cppn_impl = @import("cppn.zig");

const Solver = @import("../network/solver.zig").Solver;
const SubstrateLayout = sub_layout.SubstrateLayout;
const fastSolverGenomeFromFile = cppn_impl.fastSolverGenomeFromFile;
const checkNetworkSolverOutputs = cppn_impl.checkNetworkSolverOutputs;
const GridSubstrateLayout = sub_layout.GridSubstrateLayout;
const NodeActivationType = neat_activations.NodeActivationType;
const Options = opts.Options;
const FastNetworkLink = fast_net.FastNetworkLink;
const FastModularNetworkSolver = fast_net.FastModularNetworkSolver;
const queryCPPN = cppn_impl.queryCPPN;
const createLink = cppn_impl.createLink;
const createThresholdNormalizedLink = cppn_impl.createThresholdNormalizedLink;

/// Substrate represents substrate holding configuration of ANN with weights produced by CPPN.
/// According to HyperNEAT method, the ANN neurons are encoded as coordinates in hypercube presented
/// by this substrate. By default, neurons will be placed into substrate within grid layout
pub const Substrate = struct {
    /// The layout of neuron nodes in this substrate.
    layout: SubstrateLayout,
    /// The activation function's type for neurons encoded.
    nodes_activation: NodeActivationType,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new Substrate.
    pub fn init(allocator: std.mem.Allocator, layout: SubstrateLayout, nodes_activation: NodeActivationType) !*Substrate {
        var self = try allocator.create(Substrate);
        self.* = .{
            .allocator = allocator,
            .layout = layout,
            .nodes_activation = nodes_activation,
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *Substrate) void {
        self.layout.deinit();
        self.allocator.destroy(self);
    }

    /// Initializes a new network Solver based on current substrate layout and provided
    /// Compositional Pattern Producing Network (CPPN), which is used to define connections
    /// between network nodes. If use_leo is true, then Link Expression Output extension to
    /// the HyperNEAT will be used, instead of standard weight threshold technique of HyperNEAT,
    /// to determine whether to express link between two nodes or not. With LEO the link is
    /// expressed based on value of additional output of the CPPN (if > 0, then expressed).
    pub fn createNetworkSolver(self: *Substrate, allocator: std.mem.Allocator, cppn: Solver, use_leo: bool, options: *Options) !Solver {
        // check conditions
        if (self.layout.bias_count() > 1) {
            std.debug.print("SUBSTRATE: maximum one BIAS node per network supported\n", .{});
            return error.SubstrateExceedsMaxBiasNodes;
        }

        // the network layers will be collected in order: bias, input, output, hidden
        const first_bias: usize = 0;
        const first_input = self.layout.bias_count();
        const first_output = first_input + self.layout.inputCount();
        const first_hidden = first_output + self.layout.outputCount();
        const last_hidden = first_hidden + self.layout.hiddenCount();

        const total_neuron_count = last_hidden;

        var links = std.ArrayList(*FastNetworkLink).init(allocator);
        var bias_list = try allocator.alloc(f64, total_neuron_count);
        for (bias_list) |*v| v.* = 0;

        // give bias inputs to all hidden and output nodes.
        var coordinates = [_]f64{0} ** 4;
        var bi = first_bias;
        while (bi < first_input) : (bi += 1) {
            // the bias coordinates
            var bias_position = try self.layout.nodePosition(allocator, bi - first_bias, .BiasNeuron);
            defer bias_position.deinit();
            coordinates[0] = bias_position.x;
            coordinates[1] = bias_position.y;

            // TODO: implement GraphML XML functionality and add node to graph

            // link the bias to all hidden nodes.
            var hi = first_hidden;
            while (hi < last_hidden) : (hi += 1) {
                // get hidden neuron coordinates
                var hidden_position = try self.layout.nodePosition(allocator, hi - first_hidden, .HiddenNeuron);
                defer hidden_position.deinit();
                coordinates[2] = hidden_position.x;
                coordinates[3] = hidden_position.y;

                // TODO: implement GraphML XML functionality and add node to graph

                // find connection weight
                var link = try fastNetworkLink(allocator, &coordinates, cppn, use_leo, bi, hi, options);
                if (link != null) {
                    defer link.?.deinit();
                    bias_list[hi] = link.?.weight;

                    // TODO: implement GraphML XML functionality and add node & edge to graph
                }
            }

            // link the bias to all output nodes
            var oi = first_output;
            while (oi < first_hidden) : (oi += 1) {
                // get output neuron coordinates
                var output_pos = try self.layout.nodePosition(allocator, oi - first_output, .OutputNeuron);
                defer output_pos.deinit();
                coordinates[2] = output_pos.x;
                coordinates[3] = output_pos.y;

                // TODO: implement GraphML XML functionality and add node to graph

                // find connection weight
                var link = try fastNetworkLink(allocator, &coordinates, cppn, use_leo, bi, oi, options);
                if (link != null) {
                    defer link.?.deinit();
                    bias_list[oi] = link.?.weight;

                    // TODO: implement GraphML XML functionality and add node & edge to graph
                }
            }
        }

        if (self.layout.hiddenCount() > 0) {
            // link input nodes to hidden ones
            var in = first_input;
            while (in < first_output) : (in += 1) {
                // get coordinates of input node
                var input_pos = try self.layout.nodePosition(allocator, in - first_input, .InputNeuron);
                defer input_pos.deinit();
                coordinates[0] = input_pos.x;
                coordinates[1] = input_pos.y;

                // TODO: implement GraphML XML functionality and add node to graph

                var hi = first_hidden;
                while (hi < last_hidden) : (hi += 1) {
                    // get hidden neuron coordinates
                    var hidden_pos = try self.layout.nodePosition(allocator, hi - first_hidden, .HiddenNeuron);
                    defer hidden_pos.deinit();
                    coordinates[2] = hidden_pos.x;
                    coordinates[3] = hidden_pos.y;

                    // find connection weight
                    var link = try fastNetworkLink(allocator, &coordinates, cppn, use_leo, in, hi, options);
                    if (link != null) {
                        try links.append(link.?);

                        // TODO: implement GraphML XML functionality and add node & edge to graph
                    }
                }
            }

            // link all hidden nodes to all output nodes.
            var hi = first_hidden;
            while (hi < last_hidden) : (hi += 1) {
                var hidden_pos = try self.layout.nodePosition(allocator, hi - first_hidden, .HiddenNeuron);
                defer hidden_pos.deinit();
                coordinates[0] = hidden_pos.x;
                coordinates[1] = hidden_pos.y;

                var oi = first_output;
                while (oi < first_hidden) : (oi += 1) {
                    // get output neuron coordinates
                    var output_pos = try self.layout.nodePosition(allocator, oi - first_output, .OutputNeuron);
                    defer output_pos.deinit();
                    coordinates[2] = output_pos.x;
                    coordinates[3] = output_pos.y;

                    // find connection weight
                    var link = try fastNetworkLink(allocator, &coordinates, cppn, use_leo, hi, oi, options);
                    if (link != null) {
                        try links.append(link.?);

                        // TODO: implement GraphML XML functionality and add node & edge to graph
                    }
                }
            }
        } else {
            // connect all input nodes directly to all output nodes
            var in = first_input;
            while (in < first_output) : (in += 1) {
                // get coordinates of input node
                var input_pos = try self.layout.nodePosition(allocator, in - first_input, .InputNeuron);
                defer input_pos.deinit();
                coordinates[0] = input_pos.x;
                coordinates[1] = input_pos.y;

                // TODO: implement GraphML XML functionality and add node to graph

                var oi = first_output;
                while (oi < first_hidden) : (oi += 1) {
                    // get output neuron coordinates
                    var output_pos = try self.layout.nodePosition(allocator, oi - first_output, .OutputNeuron);
                    defer output_pos.deinit();
                    coordinates[2] = output_pos.x;
                    coordinates[3] = output_pos.y;

                    // find connection weight
                    var link = try fastNetworkLink(allocator, &coordinates, cppn, use_leo, in, oi, options);
                    if (link != null) {
                        try links.append(link.?);

                        // TODO: implement GraphML XML functionality and add node & edge to graph
                    }
                }
            }
        }

        // build activations
        var activations = try allocator.alloc(NodeActivationType, total_neuron_count);
        for (activations, 0..) |_, i| {
            activations[i] = self.activationForNeuron(i, first_output);
        }

        if (total_neuron_count == 0 or links.items.len == 0 or activations.len != total_neuron_count) {
            std.debug.print("failed to create network solver: links [{d}], nodes [{d}], activations [{d}]\n", .{ links.items.len, total_neuron_count, activations.len });
            return error.FailedToCreateNetworkSolver;
        }
        var modular_solver = try FastModularNetworkSolver.init(allocator, self.layout.bias_count(), self.layout.inputCount(), self.layout.outputCount(), total_neuron_count, activations, try links.toOwnedSlice(), bias_list, null);
        return Solver.init(modular_solver);
    }

    fn activationForNeuron(self: *Substrate, node_index: usize, first_output: usize) NodeActivationType {
        if (node_index < first_output) {
            // all bias and input neurons has null activation function associated because they actually have
            // no inputs to be activated upon
            return .NullActivation;
        } else {
            return self.nodes_activation;
        }
    }
};

fn fastNetworkLink(allocator: std.mem.Allocator, coordinates: []f64, cppn: Solver, use_leo: bool, source: usize, target: usize, options: *Options) !?*FastNetworkLink {
    var outs = try queryCPPN(allocator, coordinates, cppn);
    defer allocator.free(outs);
    if (use_leo and outs[1] > 0) {
        // add links only when CPPN LEO output signals to
        return createLink(allocator, outs[0], source, target, options.hyperneat_ctx.?.weight_range);
    } else if (!use_leo and @fabs(outs[0]) > options.hyperneat_ctx.?.link_threshold) {
        // add only links with signal exceeding provided threshold
        return createThresholdNormalizedLink(allocator, outs[0], source, target, options.hyperneat_ctx.?.link_threshold, options.hyperneat_ctx.?.weight_range);
    }
    return null;
}

test "Substrate init" {
    const allocator = std.testing.allocator;
    const bias_count: usize = 1;
    const input_count: usize = 4;
    const hidden_count: usize = 2;
    const output_count: usize = 2;
    var layout = SubstrateLayout.init(try GridSubstrateLayout.init(allocator, bias_count, input_count, hidden_count, output_count));

    // create new substrate
    var substrate = try Substrate.init(allocator, layout, .SigmoidSteepenedActivation);
    defer substrate.deinit();

    try std.testing.expect(substrate.nodes_activation == .SigmoidSteepenedActivation);
}

test "Substrate create network solver" {
    const allocator = std.testing.allocator;
    const bias_count: usize = 1;
    const input_count: usize = 4;
    const hidden_count: usize = 2;
    const output_count: usize = 2;

    var layout = SubstrateLayout.init(try GridSubstrateLayout.init(allocator, bias_count, input_count, hidden_count, output_count));

    var cppn = try fastSolverGenomeFromFile(allocator, "data/test_cppn_hyperneat_genome.json");
    defer cppn.deinit();

    var context = try Options.readFromJSON(allocator, "data/test_hyperneat.json");
    defer context.deinit();

    var substrate = try Substrate.init(allocator, layout, .SigmoidSteepenedActivation);
    defer substrate.deinit();

    var solver = try substrate.createNetworkSolver(allocator, cppn, false, context);
    defer solver.deinit();

    // test solver
    const total_node_count = bias_count + input_count + hidden_count + output_count;
    try std.testing.expect(solver.nodeCount() == total_node_count);

    const total_link_count: usize = 12;
    try std.testing.expect(solver.linkCount() == total_link_count);

    var out_expected = [_]f64{ 0.6427874813512032, 0.8685335941574246 };
    try checkNetworkSolverOutputs(allocator, solver, &out_expected, 0);
}

test "Substrate create network solver with LEO" {
    const allocator = std.testing.allocator;
    const bias_count: usize = 1;
    const input_count: usize = 4;
    const hidden_count: usize = 2;
    const output_count: usize = 2;

    var layout = SubstrateLayout.init(try GridSubstrateLayout.init(allocator, bias_count, input_count, hidden_count, output_count));

    var cppn = try fastSolverGenomeFromFile(allocator, "data/test_cppn_hyperneat_leo_genome.json");
    defer cppn.deinit();

    var context = try Options.readFromJSON(allocator, "data/test_hyperneat.json");
    defer context.deinit();

    var substrate = try Substrate.init(allocator, layout, .SigmoidSteepenedActivation);
    defer substrate.deinit();

    var solver = try substrate.createNetworkSolver(allocator, cppn, true, context);
    defer solver.deinit();

    // test solver
    const total_node_count = bias_count + input_count + hidden_count + output_count;
    try std.testing.expect(solver.nodeCount() == total_node_count);

    const total_link_count: usize = 16;
    try std.testing.expect(solver.linkCount() == total_link_count);

    var out_expected = [_]f64{ 0.5, 0.5 };
    try checkNetworkSolverOutputs(allocator, solver, &out_expected, 1e-5);
}
