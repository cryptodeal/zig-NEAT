const std = @import("std");
const zigNEAT = @import("zigNEAT");
const retina_env = @import("environment.zig");
const retina = @import("retina.zig");
const RetinaGenerationEvaluator = retina.RetinaGenerationEvaluator;

const createRetinaDataset = @import("dataset.zig").createRetinaDataset;
const Options = zigNEAT.Options;
const Genome = zigNEAT.genetics.Genome;
const Experiment = zigNEAT.experiment.Experiment;
const GenerationEvaluator = zigNEAT.experiment.GenerationEvaluator;
const Environment = retina_env.Environment;

const species_target: usize = 15;
const species_compat_adjust_freq: usize = 10;
const use_leo = true;

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    // Seed the random-number generator with current time so that
    // the numbers will be different every time we run.
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    // Load NEAT options
    var options = try Options.readFromJSON(allocator, "data/es_hyperneat.json");
    defer options.deinit();
    std.debug.print("read options sucessfully; num runs {d}\n", .{options.num_runs});

    var start_genome = try Genome.readFromJSON(allocator, "data/cppn_genome.json");
    defer start_genome.deinit();
    std.debug.print("read start_genome sucessfully\n", .{});

    // Load Modular Retina environment
    var experiment = try Experiment.init(allocator, 0);
    defer experiment.deinit();
    try experiment.trials.ensureTotalCapacityPrecise(options.num_runs);

    var env = try Environment.init(allocator, try createRetinaDataset(allocator), 4);
    defer env.deinit();

    var gen_eval = RetinaGenerationEvaluator{
        .env = env,
        .num_species_target = species_target,
        .use_leo = use_leo,
        .compat_adjust_freq = species_compat_adjust_freq,
        .allocator = allocator,
    };

    const evaluator = GenerationEvaluator.init(&gen_eval);
    try experiment.execute(allocator, rand, options, start_genome, evaluator, null);
}
