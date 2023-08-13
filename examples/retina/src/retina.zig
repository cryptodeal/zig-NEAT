const std = @import("std");
const zig_neat = @import("zigNEAT");
const retina_env = @import("environment.zig");
const createRetinaDataset = @import("dataset.zig").createRetinaDataset;

const Solver = zig_neat.network.Solver;
const NeatLogger = zig_neat.NeatLogger;
const Environment = retina_env.Environment;
const VisualObject = retina_env.VisualObject;
const MappedEvolvableSubstrateLayout = zig_neat.cppn.MappedEvolvableSubstrateLayout;
const EvolvableSubstrateLayout = zig_neat.cppn.EvolvableSubstrateLayout;
const EvolvableSubstrate = zig_neat.cppn.EvolvableSubstrate;
const Options = zig_neat.Options;
const Population = zig_neat.genetics.Population;
const Organism = zig_neat.genetics.Organism;
const Generation = zig_neat.experiment.Generation;

var logger = NeatLogger{ .log_level = std.log.Level.info };

/// Used as max value which we add error too to get an organism's fitness
pub const max_fitness: f64 = 1000;
/// The fitness value for which an organism is considered to have won the experiment
pub const fitness_threshold: f64 = max_fitness;

pub const compatability_threshold_step: f64 = 0.1;
pub const compatability_threshold_min_value: f64 = 0.3;

pub const debug = false;

pub const RetinaGenerationEvaluator = struct {
    env: *Environment,
    /// The target number of species to be maintained
    num_species_target: usize,
    /// The species compatibility threshold adjustment frequency
    compat_adjust_freq: usize,
    /// The flag to indicate if Link Expression Output should be enabled in CPPN
    use_leo: bool,

    best_fitness: f64 = 0,

    allocator: std.mem.Allocator,

    /// evaluates a population of organisms and prints their performance on the retina experiment
    pub fn generationEvaluate(self: *RetinaGenerationEvaluator, opts: *Options, pop: *Population, epoch: *Generation) !void {
        std.debug.print("best fitness: {d}\n", .{self.best_fitness});

        // Evaluate each organism on a test
        var max_population_fitness: f64 = 0.0;
        var best_link_count: usize = 0;
        var best_node_count: usize = 0;
        var best_substrate_solver: ?Solver = null;
        defer {
            if (best_substrate_solver != null) {
                best_substrate_solver.?.deinit();
            }
        }

        for (pop.organisms.items) |org| {
            var res = self.orgEval(self.allocator, opts, org) catch |err| {
                std.debug.print("failed to evaluate organism: {any}\n", .{err});
                org.fitness = -1;
                continue;
            };
            var is_winner = res.is_winner;
            var solver = res.solver;

            if (org.fitness > max_population_fitness) {
                max_population_fitness = org.fitness;
                best_link_count = @intCast(org.phenotype.?.linkCount());
                best_node_count = @intCast(org.phenotype.?.nodeCount());
                if (best_substrate_solver != null) {
                    best_substrate_solver.?.deinit();
                }
                if (org.fitness > self.best_fitness) {
                    self.best_fitness = org.fitness;
                    std.debug.print("New best fitness: {d}\n", .{org.fitness});
                    var file_name = std.ArrayList(u8).init(self.allocator);
                    try file_name.writer().print("out/best_genome_fitness_{d:.5}.json", .{org.fitness});
                    var path: []const u8 = try file_name.toOwnedSlice();
                    defer self.allocator.free(path);
                    try org.genotype.writeToJSON(self.allocator, path);
                    std.debug.print("Saved best genome to {s}\n", .{path});
                }
                best_substrate_solver = solver;
            } else {
                // free solver
                solver.deinit();
            }

            if (is_winner and (epoch.champion == null or org.fitness > epoch.champion.?.fitness)) {
                epoch.solved = true;
                epoch.winner_nodes = org.genotype.nodes.len;
                epoch.winner_genes = @as(usize, @intCast(org.genotype.extrons()));
                epoch.winner_evals = opts.pop_size * epoch.id + @as(usize, @intCast(org.genotype.id));
                epoch.champion = org;
                if (epoch.winner_nodes == 9) {
                    // TODO: dump out optimal genomes here if desired
                }
            }
        }

        // Fill statistics about current epoch
        try epoch.fillPopulationStatistics(pop);

        // TODO: Only print to file every print_every generation

        if (epoch.solved) {
            // print winner organism
            var org = epoch.champion.?;
            std.debug.print("Winner organism fitness: {d}\n", .{org.fitness});

            var depth = try org.phenotype.?.maxActivationDepthCapped(0);
            std.debug.print("Activation depth of the winner: {d}\n", .{depth});
        } else if (epoch.id < opts.num_generations - 1) {
            var species_count = pop.species.items.len;
            // adjust species count by keeping it constant
            adjustSpeciesNumber(species_count, epoch.id, self.compat_adjust_freq, self.num_species_target, opts);
            std.debug.print("{d} species -> {d} organisms [compatibility threshold: {d:.1}, target: {d}]\nbest CPNN organism [fitness: {d:.2}, links: {d}, nodes: {d}], best solver [links: {d}, nodes: {d}]", .{ species_count, pop.organisms.items.len, opts.compat_threshold, self.num_species_target, max_population_fitness, best_link_count, best_node_count, best_substrate_solver.?.linkCount(), best_substrate_solver.?.nodeCount() });
        }
    }

    const OrgEvalRes = struct {
        is_winner: bool = false,
        solver: Solver,
    };

    fn orgEval(self: *RetinaGenerationEvaluator, allocator: std.mem.Allocator, opts: *Options, organism: *Organism) !OrgEvalRes {
        var cppn_solver = Solver.init(try organism.phenotype.?.getSolver(allocator));

        // create substrate layout
        const input_count = self.env.input_size * 2; // left + right pixels of visual object
        var layout = EvolvableSubstrateLayout.init(try MappedEvolvableSubstrateLayout.init(allocator, input_count, 2));

        // create ES-HyperNEAT solver
        var substr = try EvolvableSubstrate.initWithBias(allocator, layout, opts.hyperneat_ctx.?.substrate_activator, opts.hyperneat_ctx.?.cppn_bias);
        defer substr.deinit();
        var solver = try substr.createNetworkSolver(allocator, cppn_solver, self.use_leo, opts);
        errdefer solver.deinit();
        // Evaluate the detector ANN against 256 combinations of the left and the right visual objects
        // at correct and incorrect sides of retina
        var error_sum: f64 = 0;
        var count: f64 = 0;
        var detection_error_count: f64 = 0;
        for (self.env.visual_objects) |left_object| {
            for (self.env.visual_objects) |right_object| {
                // Evaluate outputted predictions
                var loss = try self.evalNetwork(allocator, solver, left_object, right_object);
                error_sum += loss;
                count += 1;
                if (loss > 0) {
                    detection_error_count += 1;
                }
                // flush solver
                var flushed = try solver.flush();
                if (!flushed) {
                    std.debug.print("failed to flush solver after evaluation\n", .{});
                    return error.FailedToFlushSolver;
                }
            }
        }

        // Calculate the fitness score
        var fitness = max_fitness / (1 + error_sum);
        var avg_error = error_sum / count;
        var is_winner = false;
        if (fitness >= fitness_threshold) {
            is_winner = true;
            std.debug.print("Found a winner!\n", .{});
            // TODO: save solver graph to the winner organism
        }
        // Save properties to organism struct
        organism.is_winner = is_winner;
        organism.error_value = avg_error;
        organism.fitness = fitness;
        if (debug) {
            logger.info("Average error: {d}, errors sum: {d}, false detections: {d} from: {d}", .{ avg_error, error_sum, detection_error_count, count }, @src());
            logger.info("Substrate: #nodes = {d}, #edges = {d} | CPPN phenotype: #nodes = {d}, #edges = {d}", .{ solver.nodeCount(), solver.linkCount(), cppn_solver.nodeCount(), cppn_solver.linkCount() }, @src());
        }
        return .{ .is_winner = is_winner, .solver = solver };
    }

    fn evalNetwork(self: *RetinaGenerationEvaluator, allocator: std.mem.Allocator, solver: Solver, left_object: *VisualObject, right_object: *VisualObject) !f64 {
        _ = self;
        // flush current network state
        _ = solver.flush() catch return -1;

        // Create input by joining data from left and right visual objects
        var inputs = std.ArrayList(f64).init(allocator);
        defer inputs.deinit();
        try inputs.appendSlice(left_object.data);
        try inputs.appendSlice(right_object.data);

        // run evaluation
        var loss = std.math.floatMax(f64);
        solver.loadSensors(inputs.items) catch return loss;

        // Propagate activation
        var relaxed = solver.recursiveSteps() catch return loss;
        if (!relaxed) {
            std.debug.print("failed to relax network solver of the ES substrate\n", .{});
            return loss;
        }

        // get outputs and evaluate against ground truth
        var outs = try solver.readOutputs(allocator);
        defer allocator.free(outs);
        loss = try evalPredictions(allocator, outs, left_object, right_object);
        return loss;
    }
};

fn evalPredictions(allocator: std.mem.Allocator, predictions: []f64, left_obj: *VisualObject, right_obj: *VisualObject) !f64 {
    // Convert predictions[i] to 1.0 or 0.0 about 0.5 threshold
    var norm_predictions = try allocator.alloc(f64, predictions.len);
    defer allocator.free(norm_predictions);
    for (norm_predictions, 0..) |_, i| {
        if (predictions[i] >= 0.5) {
            norm_predictions[i] = 1;
        } else {
            norm_predictions[i] = 0;
            predictions[i] = 0;
        }
    }

    var targets = [_]f64{0} ** 2;

    // Set target[0] to 1.0 if LeftObj is suitable for Left side, otherwise set to 0.0
    if (left_obj.side == .LeftSide or left_obj.side == .BothSides) {
        targets[0] = 1;
    }

    // Repeat for target[1], the right side truth value
    if (right_obj.side == .RightSide or right_obj.side == .BothSides) {
        targets[1] = 1;
    }

    // Find loss as a Euclidean distance between outputs and ground truth
    var loss = (norm_predictions[0] - targets[0]) * (norm_predictions[0] * targets[0]) + (norm_predictions[1] - targets[1]) * (norm_predictions[1] - targets[1]);
    var flag: []const u8 = "match";
    if (loss != 0) {
        flag = "-";
    }

    logger.debug("[{d:.2}, {d:.2}] -> [{d:.2}, {d:.2}] '{s}'", .{ targets[0], targets[1], norm_predictions[0], norm_predictions[1], flag }, @src());
    return loss;
}

fn adjustSpeciesNumber(species_count: usize, epoch_id: usize, adjust_frequency: usize, num_species_target: usize, opts: *Options) void {
    if (@mod(epoch_id, adjust_frequency) == 0) {
        if (species_count < num_species_target) {
            opts.compat_threshold -= compatability_threshold_step;
        } else if (species_count > num_species_target) {
            opts.compat_threshold += compatability_threshold_step;
        }

        // to avoid dropping too low
        if (opts.compat_threshold < compatability_threshold_min_value) {
            opts.compat_threshold = compatability_threshold_min_value;
        }
    }
}

test "evaluate prediction" {
    const allocator = std.testing.allocator;
    var sum_loss: f64 = 0;
    const dataset = try createRetinaDataset(allocator);
    defer for (dataset) |vo| vo.deinit();
    defer allocator.free(dataset);
    for (dataset) |left_object| {
        for (dataset) |right_object| {
            var pred = [_]f64{0} ** 4;
            sum_loss += try evalPredictions(allocator, &pred, left_object, right_object);
        }
    }
    try std.testing.expect(sum_loss == 416);
}
