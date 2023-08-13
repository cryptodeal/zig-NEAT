const common = @import("common.zig");
const env = @import("environment.zig");
const std = @import("std");
const ns = @import("maze_ns.zig");
const obj = @import("maze_obj.zig");
const argsParser = @import("args");
const zig_neat = @import("zigNEAT");

const Options = zig_neat.Options;
const Genome = zig_neat.genetics.Genome;
const Environment = env.Environment;
const Experiment = zig_neat.experiment.Experiment;
const GenerationEvaluator = zig_neat.experiment.GenerationEvaluator;
const TrialRunObserver = zig_neat.experiment.TrialRunObserver;
const MazeNsGenerationEvaluator = ns.MazeNsGenerationEvaluator;
const MazeObjectiveEvaluator = obj.MazeObjectiveEvaluator;

const ExpType = enum { MazeNS, MazeOBJ };

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const cli_args = try argsParser.parseForCurrentProcess(struct {
        // This declares long options for double hyphen
        out: []const u8 = "out",
        context: []const u8 = "data/maze.neat",
        genome: []const u8 = "data/mazestartgenes",
        maze: []const u8 = "data/medium_maze.txt",
        experiment: ExpType = .MazeNS,
        timesteps: usize = 400,
        timesteps_sample: usize = 1000,
        species_target: usize = 20,
        species_adjust_freq: usize = 10,
        trials: usize = 0,
        exit_range: f64 = 5,
    }, allocator, .print);
    defer cli_args.deinit();

    std.debug.print("Loaded {s}\n", .{cli_args.options.maze});
    std.debug.print("Running {s}\n", .{@tagName(cli_args.options.experiment)});

    // Seed the random-number generator with current time so that
    // the numbers will be different every time we run.
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    // Load NEAT options
    var options = try Options.readOptions(allocator, cli_args.options.context);
    defer options.deinit();

    // Load Genome
    std.debug.print("Loading start genome for {s} experiment from file '{s}'\n", .{ @tagName(cli_args.options.experiment), cli_args.options.genome });
    var start_genome = try Genome.readFromFile(allocator, cli_args.options.genome);
    defer start_genome.deinit();

    // Load Maze environment
    var environment = try Environment.readFromFile(allocator, cli_args.options.maze);
    defer environment.deinit();
    environment.time_steps = cli_args.options.timesteps;
    environment.sample_size = cli_args.options.timesteps_sample;
    environment.exit_found_range = cli_args.options.exit_range;

    // TODO: check if output dir exists; backup to new folder with prev_name-timestamp before running experiment
    // prevents overwriting previous results

    // Override context configuration parameters with ones set from command line
    if (cli_args.options.trials > 0) {
        options.num_runs = cli_args.options.trials;
    }

    // create experiment
    var experiment = try Experiment.init(allocator, 0);
    defer experiment.deinit();
    try experiment.trials.ensureTotalCapacityPrecise(options.num_runs);

    // TODO: var obj_eval:
    var ns_eval: MazeNsGenerationEvaluator = undefined;
    var obj_eval: MazeObjectiveEvaluator = undefined;
    var evaluator: GenerationEvaluator = undefined;
    var trial_observer: TrialRunObserver = undefined;
    if (cli_args.options.experiment == .MazeNS) {
        ns_eval = MazeNsGenerationEvaluator{
            .allocator = allocator,
            .output_path = cli_args.options.out,
            .maze_env = environment,
            .num_species_target = cli_args.options.species_target,
            .compat_adjust_freq = cli_args.options.species_adjust_freq,
        };
        evaluator = GenerationEvaluator.init(&ns_eval);
        trial_observer = TrialRunObserver.init(&ns_eval);
    } else if (cli_args.options.experiment == .MazeOBJ) {
        obj_eval = MazeObjectiveEvaluator{
            .allocator = allocator,
            .output_path = cli_args.options.out,
            .maze_env = environment,
            .num_species_target = cli_args.options.species_target,
            .compat_adjust_freq = cli_args.options.species_adjust_freq,
        };
        evaluator = GenerationEvaluator.init(&obj_eval);
        trial_observer = TrialRunObserver.init(&obj_eval);
    } else {
        std.debug.print("Unsupported experiment name requested: {s}\n", .{@tagName(cli_args.options.experiment)});
    }

    try experiment.execute(allocator, rand, options, start_genome, evaluator, trial_observer);
}

test {
    _ = common;
    _ = env;
}
