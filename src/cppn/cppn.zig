const std = @import("std");
const fast_net = @import("../network/fast_network.zig");
const quad_tree = @import("quad_tree.zig");
const es_layout = @import("evolvable_substrate_layout.zig");
const layout = @import("substrate_layout.zig");

const Genome = @import("../genetics/genome.zig").Genome;
const Solver = @import("../network/solver.zig").Solver;
const FastNetworkLink = fast_net.FastNetworkLink;

// exports
/// Defines a hashable value for a given PointF instance
pub const PointFHash = quad_tree.PointFHash;
/// Defines a point with f64 precision coordinates.
pub const PointF = quad_tree.PointF;
/// Defines the quad-point in the 4 dimensional hypercube.
pub const QuadPoint = quad_tree.QuadPoint;
/// Defines a quad-tree node to model 4 dimensional hypercube.
pub const QuadNode = quad_tree.QuadNode;
pub const EvolvableSubstrate = @import("evolvable_substrate.zig").EvolvableSubstrate;
pub const EvolvableSubstrateLayout = es_layout.EvolvableSubstrateLayout;
pub const MappedEvolvableSubstrateLayout = es_layout.MappedEvolvableSubstrateLayout;
pub const SubstrateLayout = layout.SubstrateLayout;
pub const GridSubstrateLayout = layout.GridSubstrateLayout;
pub const Substrate = @import("substrate.zig").Substrate;

/// Reads CPPN from specified Genome and initializes Solver.
pub fn fastSolverGenomeFromFile(allocator: std.mem.Allocator, path: []const u8) !Solver {
    const genome = try Genome.readFromJSON(allocator, path);
    defer genome.deinit();
    _ = try genome.genesis(allocator, genome.id);
    return genome.phenotype.?.fastNetworkSolver(allocator);
}

/// Creates normalized by threshold value link between source and target nodes, given calculated CPPN output for their coordinates.
pub fn createThresholdNormalizedLink(allocator: std.mem.Allocator, cppn_output: f64, src_idx: usize, dst_idx: usize, link_threshold: f64, weight_range: f64) !*FastNetworkLink {
    var weight = (@fabs(cppn_output) - link_threshold) / (1 - link_threshold); // normalize [0, 1]
    weight *= weight_range; // scale to fit given weight range
    if (std.math.signbit(cppn_output)) {
        weight *= -1;
    }
    return FastNetworkLink.init(allocator, src_idx, dst_idx, weight);
}

/// Creates link between source and target nodes, given calculated CPPN output for their coordinates.
pub fn createLink(allocator: std.mem.Allocator, cppn_output: f64, src_idx: usize, dst_idx: usize, weight_range: f64) !*FastNetworkLink {
    var weight = cppn_output;
    weight *= weight_range; // scale to fit given weight range
    return FastNetworkLink.init(allocator, src_idx, dst_idx, weight);
}

/// Calculates outputs of provided CPPN network Solver with given hypercube coordinates.
pub fn queryCPPN(allocator: std.mem.Allocator, coordinates: []f64, cppn: Solver) ![]f64 {
    // flush networks activation from previous run
    var res = try cppn.flush();
    if (!res) {
        std.debug.print("failed to flush CPPN network\n", .{});
        return error.FlushFailed;
    }
    // load inputs
    try cppn.loadSensors(coordinates);
    // do activations
    res = try cppn.recursiveSteps();
    if (!res) {
        std.debug.print("failed to relax CPPN network recursively\n", .{});
        return error.RecursiveActivationFailed;
    }
    return cppn.readOutputs(allocator);
}

/// Determines variance among CPPN values for certain hypercube region around specified node.
/// This variance is a heuristic indicator of the heterogeneity (i.e. presence of information) of a region.
pub fn nodeVariance(allocator: std.mem.Allocator, node: *QuadNode) !f64 {
    // quick check
    if (node.nodes.items.len == 0) {
        return 0;
    }

    var cppn_vals = try nodeCPPNValues(allocator, node);
    defer allocator.free(cppn_vals);

    // calculate median and variance
    var mean_w: f64 = 0;
    var variance: f64 = 0;
    for (cppn_vals) |w| {
        mean_w += w;
    }
    mean_w /= @as(f64, @floatFromInt(cppn_vals.len));

    for (cppn_vals) |w| {
        variance += std.math.pow(f64, w - mean_w, 2);
    }
    variance /= @as(f64, @floatFromInt(cppn_vals.len));

    return variance;
}

/// Collects the CPPN values stored in a given quadtree node. Used to
/// estimate the variance in a certain region of space around node
pub fn nodeCPPNValues(allocator: std.mem.Allocator, n: *QuadNode) ![]f64 {
    if (n.nodes.items.len > 0) {
        var accumulator = std.ArrayList(f64).init(allocator);
        for (n.nodes.items) |p| {
            // go into child nodes
            var p_vals = try nodeCPPNValues(allocator, p);
            defer allocator.free(p_vals);
            try accumulator.appendSlice(p_vals);
        }
        return accumulator.toOwnedSlice();
    } else {
        var res = try allocator.alloc(f64, 1);
        res[0] = n.weight();
        return res;
    }
}

/// Used to hash float values (asserts non NaN)
pub fn normalizedFloatHash(hasher: anytype, key: anytype) void {
    const info = @typeInfo(@TypeOf(key));
    if (info != .Float) @compileError("only float types are allowed");
    std.debug.assert(!std.math.isNan(key));
    var norm_key = if (key == 0.0) 0.0 else key;
    std.hash.autoHash(hasher, @as(std.meta.Int(.unsigned, info.Float.bits), @bitCast(norm_key)));
}

// test utils/tests

fn buildTree(allocator: std.mem.Allocator) !*QuadNode {
    var root = try QuadNode.init(allocator, 0, 0, 1, 1);
    try root.nodes.append(try QuadNode.init(allocator, -1, 1, 0.5, 2));
    try root.nodes.append(try QuadNode.init(allocator, -1, -1, 0.5, 2));
    try root.nodes.append(try QuadNode.init(allocator, 1, 1, 0.5, 2));
    try root.nodes.append(try QuadNode.init(allocator, 1, -1, 0.5, 2));
    fillW(root.nodes.items, 2);
    try root.nodes.items[0].nodes.append(try QuadNode.init(allocator, -1, 1, 0.5, 3));
    try root.nodes.items[0].nodes.append(try QuadNode.init(allocator, -1, -1, 0.5, 3));
    try root.nodes.items[0].nodes.append(try QuadNode.init(allocator, 1, 1, 0.5, 3));
    try root.nodes.items[0].nodes.append(try QuadNode.init(allocator, 1, -1, 0.5, 3));
    fillW(root.nodes.items[0].nodes.items, 1);
    return root;
}

fn fillW(nodes: []*QuadNode, factor: f64) void {
    for (nodes, 0..) |n, i| {
        n.cppn_out[0] = @as(f64, @floatFromInt(i)) * factor;
    }
}

pub fn checkNetworkSolverOutputs(allocator: std.mem.Allocator, solver: Solver, out_expected: []f64, delta: f64) !void {
    var signals = [_]f64{ 0.9, 5.2, 1.2, 0.6 };
    try solver.loadSensors(&signals);
    var res = try solver.recursiveSteps();
    try std.testing.expect(res);
    var outs = try solver.readOutputs(allocator);
    defer allocator.free(outs);
    for (outs, 0..) |v, i| {
        try std.testing.expectApproxEqAbs(out_expected[i], v, delta);
    }
}

test "QuadNode node variance" {
    const allocator = std.testing.allocator;
    var root = try buildTree(allocator);
    defer root.deinit();

    // get variance and check results
    var variance = try nodeVariance(allocator, root);
    try std.testing.expectApproxEqAbs(@as(f64, 3.3877551020408165), variance, 1e-16);
}

test "QuadNode node CPPN values" {
    const allocator = std.testing.allocator;
    var root = try buildTree(allocator);
    defer root.deinit();

    // get CPPN values and test results
    var vals = try nodeCPPNValues(allocator, root);
    defer allocator.free(vals);

    try std.testing.expect(vals.len == 7);

    const expected = [_]f64{ 0, 1, 2, 3, 2, 4, 6 };
    try std.testing.expectEqualSlices(f64, &expected, vals);
}

test "Solver from Genome file" {
    const allocator = std.testing.allocator;
    var solver = try fastSolverGenomeFromFile(allocator, "data/test_cppn_hyperneat_genome.json");
    defer solver.deinit();
    try std.testing.expect(solver.nodeCount() == 7);
    try std.testing.expect(solver.linkCount() == 7);

    // test query
    var coords = [_]f64{ 0, 0, 0.5, 0.5 };
    var outs = try queryCPPN(allocator, &coords, solver);
    defer allocator.free(outs);
    try std.testing.expect(outs.len == 1);
    try std.testing.expectApproxEqAbs(@as(f64, 1e-16), outs[0], @as(f64, 0.4864161653290716));
}

test {
    std.testing.refAllDecls(@This());
}
