const std = @import("std");
const zig_neat = @import("zigNEAT");

const NeatLogger = zig_neat.NeatLogger;
const Population = zig_neat.genetics.Population;
const Organism = zig_neat.genetics.Organism;
const Options = zig_neat.Options;
const EpochExecutorType = zig_neat.EpochExecutorType;
const GenomeCompatibilityMethod = zig_neat.GenomeCompatibilityMethod;
const GenerationEvaluator = zig_neat.experiment.GenerationEvaluator;
const Genome = zig_neat.genetics.Genome;
const Experiment = zig_neat.experiment.Experiment;
const Generation = zig_neat.experiment.Generation;
const NodeActivationType = zig_neat.math.NodeActivationType;

var logger = NeatLogger{ .log_level = std.log.Level.info };

const fitness_threshold: f64 = 15.5;

const XorGenerationEvaluator = struct {
    pub fn generationEvaluate(self: *XorGenerationEvaluator, opts: *Options, pop: *Population, epoch: *Generation) !void {
        // Evaluate each organism on a test
        for (pop.organisms.items) |org| {
            var res = try self.orgEval(org);

            if (res and (epoch.champion == null or org.fitness > epoch.champion.?.fitness)) {
                epoch.solved = true;
                epoch.winner_nodes = org.genotype.nodes.len;
                epoch.winner_genes = @as(usize, @intCast(org.genotype.extrons()));
                epoch.winner_evals = opts.pop_size * epoch.id + @as(usize, @intCast(org.genotype.id));
                epoch.champion = org;
                // TODO: add functionality to write genome as JSON
                //if (epoch.winner_nodes == 5) {
                // You could dump out optimal genomes here if desired
                //}
            }
        }

        // Fill statistics about current epoch
        try epoch.fillPopulationStatistics(pop);

        // TODO: Only print to file every print_every generation

        if (epoch.solved) {
            // print winner organism
            var org: *Organism = epoch.champion.?;
            std.debug.print("Winner organism fitness: {d}\n", .{org.fitness});

            var depth = try org.phenotype.?.maxActivationDepthCapped(0);
            std.debug.print("Activation depth of the winner: {d}\n", .{depth});

            // TODO: write winner's genome to file (not implemented yet)
        }
    }

    pub fn orgEval(_: *XorGenerationEvaluator, organism: *Organism) !bool {
        // The four possible input combinations to xor
        // The first number is for biasing
        var in = [4][3]f64{
            [3]f64{ 1.0, 0.0, 0.0 },
            [3]f64{ 1.0, 0.0, 1.0 },
            [3]f64{ 1.0, 1.0, 0.0 },
            [3]f64{ 1.0, 1.0, 1.0 },
        };

        // The max depth of the network to be activated
        var net_depth = try organism.phenotype.?.maxActivationDepthCapped(0);
        if (net_depth == 0) {
            logger.err("Network depth: {d} for organism: {d}", .{ net_depth, organism.genotype.id }, @src());
            return false;
        }

        // Check for successful activation
        var success = false;
        // The four outputs
        var out: [4]f64 = undefined;

        // Load and activate the network on each input
        var count: usize = 0;
        while (count < 4) : (count += 1) {
            var input = in[count];
            organism.phenotype.?.loadSensors(&input);

            // Use depth to ensure full relaxation
            success = organism.phenotype.?.forwardSteps(net_depth) catch {
                logger.err("Failed to activate network at call to `forwardSteps`", .{}, @src());
                return false;
            };

            out[count] = organism.phenotype.?.outputs[0].activation;

            // Flush network for subsequent use
            _ = try organism.phenotype.?.flush();
        }

        if (success) {
            // Mean Squared Error
            var error_sum: f64 = @fabs(out[0]) + @fabs(1 - out[1]) + @fabs(1 - out[2]) + @fabs(out[3]); // ideal == 0
            var target: f64 = 4 - error_sum;
            organism.fitness = std.math.pow(f64, 4 - error_sum, 2);
            organism.error_value = std.math.pow(f64, 4 - target, 2);
        } else {
            // The network is flawed (shouldn't happen) - flag as anomaly
            organism.error_value = 1.0;
            organism.fitness = 0.0;
        }

        if (organism.fitness > fitness_threshold) {
            organism.is_winner = true;
            std.debug.print(">>>> Output activations: {any}\n", .{out});
        } else {
            organism.is_winner = false;
        }

        return organism.is_winner;
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    const allocator = arena.allocator();
    var opts: *Options = try Options.readOptions(allocator, "data/basic_xor.neat");
    defer opts.deinit();

    // initialize Genome from file
    var start_genome = try Genome.readFromFile(allocator, "data/xorstartgenes");
    defer start_genome.deinit();

    var experiment = try Experiment.init(allocator, 0);
    defer experiment.deinit();
    try experiment.trials.ensureTotalCapacityPrecise(opts.num_runs);

    var xor_evaluator = XorGenerationEvaluator{};

    const evaluator = GenerationEvaluator.init(&xor_evaluator);

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    try experiment.execute(allocator, rand, opts, start_genome, evaluator, null);

    // var res = experiment.avgWinnerStats();
    var avg_epoch_duration = experiment.avgEpochDuration();
    std.debug.print("avg_epoch_duration: {d}\n", .{avg_epoch_duration});
}
