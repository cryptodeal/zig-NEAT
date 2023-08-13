const std = @import("std");
const neat_activations = @import("../math/activations.zig");
const es_layout = @import("evolvable_substrate_layout.zig");
const opts = @import("../opts.zig");
const fast_net = @import("../network/fast_network.zig");
const cppn_impl = @import("cppn.zig");
const quad = @import("quad_tree.zig");

const Solver = @import("../network/solver.zig").Solver;
const QuadNode = quad.QuadNode;
const NodeActivationType = neat_activations.NodeActivationType;
const QuadPoint = quad.QuadPoint;
const PointF = quad.PointF;
const check_network_solver_outputs = cppn_impl.check_network_solver_outputs;
const fastSolverGenomeFromFile = cppn_impl.fastSolverGenomeFromFile;
const createLink = cppn_impl.createLink;
const createThresholdNormalizedLink = cppn_impl.createThresholdNormalizedLink;
const FastNetworkLink = fast_net.FastNetworkLink;
const FastModularNetworkSolver = fast_net.FastModularNetworkSolver;
const EvolvableSubstrateLayout = es_layout.EvolvableSubstrateLayout;
const MappedEvolvableSubstrateLayout = es_layout.MappedEvolvableSubstrateLayout;
const Options = opts.Options;
const nodeVariance = cppn_impl.nodeVariance;
const Genome = @import("../genetics/genome.zig").Genome;

/// The evolvable substrate holds configuration of ANN produced by CPPN within hypecube where each 4-dimensional point
/// mark connection weight between two ANN units. The topology of ANN is not rigid as in plain substrate and can be evolved
/// by introducing novel nodes to the ANN. This provides extra benefits that the topology of ANN should not be handcrafted
/// by human, but produced during substrate generation from controlling CPPN and nodes locations may be arbitrary that suits
/// the best for the task at hand.
pub const EvolvableSubstrate = struct {
    /// The layout of neuron nodes in this substrate
    layout: EvolvableSubstrateLayout,
    /// The activation function's type for neurons encoded
    nodes_activation: NodeActivationType,

    /// The CPPN network solver to describe geometry of substrate
    cppn: Solver = undefined,
    /// The reusable coordinates buffer
    coords: []f64,

    allocator: std.mem.Allocator,

    /// Creates new instance of evolvable substrate
    pub fn init(allocator: std.mem.Allocator, layout: EvolvableSubstrateLayout, nodes_activation: NodeActivationType) !*EvolvableSubstrate {
        var self = try allocator.create(EvolvableSubstrate);
        var coords = try allocator.alloc(f64, 4);
        for (coords) |*v| v.* = 0;
        self.* = .{
            .allocator = allocator,
            .layout = layout,
            .nodes_activation = nodes_activation,
            .coords = coords,
        };
        return self;
    }

    pub fn initWithBias(allocator: std.mem.Allocator, layout: EvolvableSubstrateLayout, nodes_activation: NodeActivationType, cppn_bias: f64) !*EvolvableSubstrate {
        var self = try allocator.create(EvolvableSubstrate);
        var coords = try allocator.alloc(f64, 5);
        for (coords, 0..) |*v, i| v.* = if (i == 0) cppn_bias else 0;
        self.* = .{
            .allocator = allocator,
            .layout = layout,
            .nodes_activation = nodes_activation,
            .coords = coords,
        };
        return self;
    }

    /// frees memory associated with the evolvable substrate
    pub fn deinit(self: *EvolvableSubstrate) void {
        self.layout.deinit();
        self.cppn.deinit();
        self.allocator.free(self.coords);
        self.allocator.destroy(self);
    }

    pub fn createNetworkSolver(self: *EvolvableSubstrate, allocator: std.mem.Allocator, cppn: Solver, use_leo: bool, context: *Options) !Solver {
        self.cppn = cppn;

        // the network layers will be collected in order: bias, input, output, hidden
        var first_input: usize = 0;
        var first_output: usize = first_input + self.layout.inputCount();
        var first_hidden: usize = first_output + self.layout.outputCount();

        var links = std.ArrayList(*FastNetworkLink).init(allocator);
        // The map to hold already created connections
        var conn_map = std.StringHashMap(*FastNetworkLink).init(allocator);
        defer {
            var iterator = conn_map.keyIterator();
            while (iterator.next()) |key| {
                allocator.free(key.*);
            }
            conn_map.deinit();
        }

        // Build connections from input nodes to the hidden nodes
        var root: *QuadNode = undefined;
        var in = first_input;
        while (in < first_output) : (in += 1) {
            // Analyze outgoing connectivity pattern from this input
            var input = try self.layout.nodePosition(allocator, in - first_input, .InputNeuron);
            defer input.deinit();

            // add input node to graph
            // TODO: implement GraphML XML functionality and add node to graph
            root = try self.divideAndInit(allocator, input.x, input.y, true, context);
            defer root.deinit();

            var q_points = try self.pruneAndExpress(allocator, input.x, input.y, root, true, context);
            defer {
                for (q_points) |qp| qp.deinit();
                allocator.free(q_points);
            }

            // iterate over quad points and add nodes/connections
            for (q_points) |qp| {
                // add hidden node to the substrate layout if needed
                var target_idx = try self.addHiddenNode(allocator, qp, first_hidden);

                // if (missing_node) {
                // add a node to the graph
                // TODO: implement GraphML XML functionality and add node to graph
                // }

                // add connection
                _ = try self.addLink(allocator, use_leo, &conn_map, &links, qp, in, target_idx, context);
                // if (link != null) {
                // add an edge to the graph
                // TODO: implement GraphML XML functionality and add edge to graph
                // }
            }
        }

        // Build more hidden nodes into unexplored area through a number of iterations
        var first_hidden_iter = first_hidden;
        var last_hidden: i64 = @as(i64, @intCast(first_hidden_iter + self.layout.hiddenCount()));
        var step: usize = 0;
        while (step < context.es_hyperneat_ctx.?.es_iterations) : (step += 1) {
            var hi = @as(i64, @intCast(first_hidden_iter));
            while (hi < last_hidden) : (hi += 1) {
                // Analyze outgoing connectivity pattern from this hidden node
                var hidden = try self.layout.nodePosition(allocator, @as(usize, @intCast(hi)) - first_hidden, .HiddenNeuron);

                root = try self.divideAndInit(allocator, hidden.x, hidden.y, true, context);
                defer root.deinit();

                var q_points = try self.pruneAndExpress(allocator, hidden.x, hidden.y, root, true, context);
                defer {
                    for (q_points) |qp| qp.deinit();
                    allocator.free(q_points);
                }

                // iterate over quad points and add nodes/connections
                for (q_points) |qp| {
                    // add hidden node to the substrate layout if needed
                    var target_idx = try self.addHiddenNode(allocator, qp, first_hidden);

                    // if (missing_node) {
                    // add a node to the graph
                    // TODO: implement GraphML XML functionality and add node to graph
                    // }

                    // add connection
                    _ = try self.addLink(allocator, use_leo, &conn_map, &links, qp, @as(usize, @intCast(hi)), target_idx, context);
                    // if (link != null) {
                    // add an edge to the graph
                    // TODO: implement GraphML XML functionality and add edge to graph
                    //}
                }
            }
            // move to the next window
            first_hidden_iter = @as(usize, @intCast(last_hidden));
            last_hidden = last_hidden + (@as(i64, @intCast(self.layout.hiddenCount())) - last_hidden);
        }

        // Connect hidden nodes to the output
        var oi: usize = first_output;
        while (oi < first_hidden) : (oi += 1) {
            // Analyze incoming connectivity pattern
            var output = try self.layout.nodePosition(allocator, oi - first_output, .OutputNeuron);
            defer output.deinit();

            // add output node to graph
            // TODO: implement GraphML XML functionality and add node to graph

            root = try self.divideAndInit(allocator, output.x, output.y, false, context);
            defer root.deinit();

            var q_points = try self.pruneAndExpress(allocator, output.x, output.y, root, false, context);
            defer {
                for (q_points) |qp| qp.deinit();
                allocator.free(q_points);
            }

            // iterate over quad points and add nodes/connections where appropriate
            for (q_points) |qp| {
                var node_point = try PointF.init(allocator, qp.x1, qp.y1);
                defer node_point.deinit();
                var source_idx = self.layout.indexOfHidden(node_point) catch continue;
                source_idx += first_hidden; // adjust index to the global indexes space

                // add connection
                _ = try self.addLink(allocator, use_leo, &conn_map, &links, qp, source_idx, oi, context);
                // if (link != null) {
                // add an edge to the graph
                // TODO: implement GraphML XML functionality and add edge to graph
                //}
            }
        }

        var total_neuron_count = self.layout.inputCount() + self.layout.outputCount() + self.layout.hiddenCount();
        // build activations
        var activations = try allocator.alloc(NodeActivationType, total_neuron_count);
        var i: usize = 0;
        while (i < total_neuron_count) : (i += 1) {
            if (i < first_output) {
                // input nodes - NULL activation
                activations[i] = .NullActivation;
            } else {
                // hidden/output nodes - defined activation
                activations[i] = self.nodes_activation;
            }
        }

        // create fast network solver
        var modular_solver = try FastModularNetworkSolver.init(allocator, 0, self.layout.inputCount(), self.layout.outputCount(), total_neuron_count, activations, try links.toOwnedSlice(), null, null);
        return Solver.init(modular_solver);
    }

    fn addHiddenNode(self: *EvolvableSubstrate, allocator: std.mem.Allocator, qp: *QuadPoint, first_hidden: usize) !usize {
        var node_point = try PointF.init(allocator, qp.x2, qp.y2);
        var added = false;
        var target_idx = self.layout.indexOfHidden(node_point) catch blk: {
            // add hidden node to the substrate layout
            var temp_idx = try self.layout.addHiddenNode(node_point);
            temp_idx += first_hidden;
            added = true;
            break :blk temp_idx;
        };
        if (!added) {
            target_idx += first_hidden;
            node_point.deinit();
        }
        return target_idx;
    }

    /// The function to add new connection if appropriate while constructing network solver
    fn addLink(_: *EvolvableSubstrate, allocator: std.mem.Allocator, use_leo: bool, conn_map: *std.StringHashMap(*FastNetworkLink), connections: *std.ArrayList(*FastNetworkLink), qp: *QuadPoint, source: usize, target: usize, context: *Options) !?*FastNetworkLink {
        var key_list = std.ArrayList(u8).init(allocator);
        try key_list.writer().print("{d}_{d}", .{ source, target });
        var key = try key_list.toOwnedSlice();
        if (conn_map.contains(key)) {
            allocator.free(key);
            return null;
        }
        var link: ?*FastNetworkLink = null;
        if (use_leo and qp.cppn_out[1] > 0) {
            link = try createLink(allocator, qp.weight(), source, target, context.hyperneat_ctx.?.weight_range);
        } else if (!use_leo and @fabs(qp.weight()) > context.hyperneat_ctx.?.link_threshold) {
            // add only connections with signal exceeding provided threshold
            link = try createThresholdNormalizedLink(allocator, qp.weight(), source, target, context.hyperneat_ctx.?.link_threshold, context.hyperneat_ctx.?.weight_range);
        }
        if (link != null) {
            try connections.append(link.?);
            try conn_map.put(key, link.?);
            return link;
        } else {
            allocator.free(key);
            return null;
        }
    }

    fn divideAndInit(self: *EvolvableSubstrate, allocator: std.mem.Allocator, a: f64, b: f64, outgoing: bool, context: *Options) !*QuadNode {
        var root = try QuadNode.init(allocator, 0, 0, 1, 1);
        const queue_type = std.TailQueue(*QuadNode);
        var queue = queue_type{};
        var init_node = try allocator.create(queue_type.Node);
        init_node.* = .{ .data = root };
        queue.append(init_node);
        while (queue.len > 0) {
            // de-queue
            var queue_node = queue.popFirst();
            defer allocator.destroy(queue_node.?);
            var p: *QuadNode = queue_node.?.data;

            // Divide into sub-regions and assign children to parent
            try p.nodes.append(try QuadNode.init(allocator, p.x - p.width / 2, p.y - p.width / 2, p.width / 2, p.level + 1));
            try p.nodes.append(try QuadNode.init(allocator, p.x - p.width / 2, p.y + p.width / 2, p.width / 2, p.level + 1));
            try p.nodes.append(try QuadNode.init(allocator, p.x + p.width / 2, p.y + p.width / 2, p.width / 2, p.level + 1));
            try p.nodes.append(try QuadNode.init(allocator, p.x + p.width / 2, p.y - p.width / 2, p.width / 2, p.level + 1));

            for (p.nodes.items) |node| {
                if (outgoing) {
                    // Querying connection from input or hidden node (Outgoing connectivity pattern)
                    allocator.free(node.cppn_out);
                    node.cppn_out = try self.queryCPPN(allocator, a, b, node.x, node.y);
                } else {
                    // Querying connection to output node (Incoming connectivity pattern)
                    allocator.free(node.cppn_out);
                    node.cppn_out = try self.queryCPPN(allocator, node.x, node.y, a, b);
                }
            }

            // Divide until initial resolution or if variance is still high
            if (p.level < context.es_hyperneat_ctx.?.initial_depth or (p.level < context.es_hyperneat_ctx.?.maximal_depth and try nodeVariance(allocator, p) > context.es_hyperneat_ctx.?.division_threshold)) {
                for (p.nodes.items) |c| {
                    var node = try allocator.create(queue_type.Node);
                    node.* = .{ .data = c };
                    queue.append(node);
                }
            }
        }

        return root;
    }

    /// Decides what regions should have higher neuron density based on variation and express new neurons and connections into
    /// these regions.
    /// Receives coordinates of source (outgoing = true) or target node (outgoing = false) at (a, b) and initialized quadtree node.
    /// Adds the connections that are in bands of the two-dimensional cross-section of the  hypercube containing the source
    /// or target node to the connections list and return modified list.
    fn pruneAndExpress(self: *EvolvableSubstrate, allocator: std.mem.Allocator, a: f64, b: f64, node: *QuadNode, outgoing: bool, context: *Options) ![]*QuadPoint {
        var connections = std.ArrayList(*QuadPoint).init(allocator);
        // fast check
        if (node.nodes.items.len == 0) return connections.toOwnedSlice();

        // Traverse quadtree depth-first until the current node’s variance is smaller than the variance threshold or
        // until the node has no children (which means that the variance is zero).
        var left: f64 = 0;
        var right: f64 = 0;
        var top: f64 = 0;
        var bottom: f64 = 0;
        for (node.nodes.items) |quad_node| {
            var child_variance = try nodeVariance(allocator, quad_node);
            if (child_variance >= context.es_hyperneat_ctx.?.variance_threshold) {
                var new_cxns = try self.pruneAndExpress(allocator, a, b, quad_node, outgoing, context);
                defer allocator.free(new_cxns);
                try connections.appendSlice(new_cxns);
            } else {
                // Determine if point is in a band by checking neighbor CPPN values
                if (outgoing) {
                    var l = try self.queryCPPN(allocator, a, b, quad_node.x - node.width, quad_node.y);
                    defer allocator.free(l);
                    left = @fabs(quad_node.weight() - l[0]);
                    var r = try self.queryCPPN(allocator, a, b, quad_node.x + node.width, quad_node.y);
                    defer allocator.free(r);
                    right = @fabs(quad_node.weight() - r[0]);
                    var t = try self.queryCPPN(allocator, a, b, quad_node.x, quad_node.y - node.width);
                    defer allocator.free(t);
                    top = @fabs(quad_node.weight() - t[0]);
                    var bt = try self.queryCPPN(allocator, a, b, quad_node.x, quad_node.y + node.width);
                    defer allocator.free(bt);
                    bottom = @fabs(quad_node.weight() - bt[0]);
                } else {
                    var l = try self.queryCPPN(allocator, quad_node.x - node.width, quad_node.y, a, b);
                    defer allocator.free(l);
                    left = @fabs(quad_node.weight() - l[0]);
                    var r = try self.queryCPPN(allocator, quad_node.x + node.width, quad_node.y, a, b);
                    defer allocator.free(r);
                    right = @fabs(quad_node.weight() - r[0]);
                    var t = try self.queryCPPN(allocator, quad_node.x, quad_node.y - node.width, a, b);
                    defer allocator.free(t);
                    top = @fabs(quad_node.weight() - t[0]);
                    var bt = try self.queryCPPN(allocator, quad_node.x, quad_node.y + node.width, a, b);
                    defer allocator.free(bt);
                    bottom = @fabs(quad_node.weight() - bt[0]);
                }
                if (@max(@min(top, bottom), @min(left, right)) > context.es_hyperneat_ctx.?.banding_threshold) {
                    // Create new connection specified by QuadPoint(x1,y1,x2,y2,weight) in 4D hypercube
                    var conn: *QuadPoint = undefined;
                    if (outgoing) {
                        conn = try QuadPoint.init(allocator, a, b, quad_node.x, quad_node.y, quad_node);
                    } else {
                        conn = try QuadPoint.init(allocator, quad_node.x, quad_node.y, a, b, quad_node);
                    }
                    try connections.append(conn);
                }
            }
        }
        return connections.toOwnedSlice();
    }

    fn queryCPPN(self: *EvolvableSubstrate, allocator: std.mem.Allocator, x1: f64, y1: f64, x2: f64, y2: f64) ![]f64 {
        var offset: usize = 0;
        if (self.coords.len == 5) {
            // CPPN bias defined
            offset = 1;
        }
        self.coords[offset] = x1;
        self.coords[offset + 1] = y1;
        self.coords[offset + 2] = x2;
        self.coords[offset + 3] = y2;
        return cppn_impl.queryCPPN(allocator, self.coords, self.cppn);
    }
};

const cppn_hyperneat_test_genome_path = "data/test_cppn_hyperneat_genome.json";

test "EvolvableSubstrate create network solver" {
    const allocator = std.testing.allocator;
    const input_count: usize = 4;
    const output_count: usize = 2;

    var layout = EvolvableSubstrateLayout.init(try MappedEvolvableSubstrateLayout.init(allocator, input_count, output_count));

    var substr = try EvolvableSubstrate.init(allocator, layout, .SigmoidSteepenedActivation);
    defer substr.deinit();

    const cppn = try fastSolverGenomeFromFile(allocator, cppn_hyperneat_test_genome_path);

    var context = try Options.readFromJSON(allocator, "data/test_es_hyperneat.json");
    defer context.deinit();

    // test solver creation
    var solver = try substr.createNetworkSolver(allocator, cppn, false, context);
    defer solver.deinit();

    var total_node_count = input_count + output_count + layout.hiddenCount();
    try std.testing.expect(total_node_count == solver.nodeCount());
    try std.testing.expect(solver.linkCount() == 8);

    var out_expected = [_]f64{ 0.5, 0.5 };
    try check_network_solver_outputs(allocator, solver, &out_expected, 1e-8);
}

test "EvolvableSubstrate create network solver LEO" {
    const allocator = std.testing.allocator;
    const input_count: usize = 4;
    const output_count: usize = 2;

    var layout = EvolvableSubstrateLayout.init(try MappedEvolvableSubstrateLayout.init(allocator, input_count, output_count));

    var substr = try EvolvableSubstrate.init(allocator, layout, .SigmoidSteepenedActivation);
    defer substr.deinit();

    const cppn = try fastSolverGenomeFromFile(allocator, "data/test_cppn_hyperneat_leo_genome.json");

    var context = try Options.readFromJSON(allocator, "data/test_es_hyperneat.json");
    defer context.deinit();

    // test solver creation
    var solver = try substr.createNetworkSolver(allocator, cppn, true, context);
    defer solver.deinit();

    var total_node_count = input_count + output_count + layout.hiddenCount();
    try std.testing.expect(total_node_count == solver.nodeCount());
    try std.testing.expect(solver.linkCount() == 19);

    var out_expected = [_]f64{ 0.5, 0.5 };
    try check_network_solver_outputs(allocator, solver, &out_expected, 1e-8);
}
