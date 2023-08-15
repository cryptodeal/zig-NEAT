const std = @import("std");
const pop_epoch = @import("../genetics/population_epoch.zig");
const neat_options = @import("../opts.zig");
const neat_population = @import("../genetics/population.zig");
const exp_generation = @import("generation.zig");
const experiment = @import("experiment.zig");
const trial = @import("trial.zig");
const utils = @import("utils.zig");

// exports

pub const Experiment = experiment.Experiment;
/// Holds the average number of nodes, genes, organisms evaluations,
/// and species diversity of winners among all trials in the experiment.
pub const AvgWinnerStats = experiment.AvgWinnerStats;
/// Utility functions for statistical analysis of floating point slices.
pub const floats = @import("floats.zig");
pub const Generation = exp_generation.Generation;
pub const GenerationAvg = exp_generation.GenerationAvg;
pub const Trial = trial.Trial;
pub const WinnerStats = trial.WinnerStats;
pub const TrialAvg = trial.TrialAvg;
pub const createOutDirForTrial = utils.createOutDirForTrial;
pub const TrialRunObserver = @import("trial_run_observer.zig");
pub const GenerationEvaluator = @import("generation_evaluator.zig");

const SequentialPopulationEpochExecutor = pop_epoch.SequentialPopulationEpochExecutor;
const ParallelPopulationEpochExecutor = pop_epoch.ParallelPopulationEpochExecutor;
const Options = neat_options.Options;
const EpochExecutorType = neat_options.EpochExecutorType;
const Population = neat_population.Population;

/// Union of Epoch Executors (Generics).
pub const EpochExecutor = union(enum) {
    sequential: *SequentialPopulationEpochExecutor,
    parallel: *ParallelPopulationEpochExecutor,

    pub fn deinit(self: EpochExecutor) void {
        switch (self) {
            inline else => |s| s.deinit(),
        }
    }

    pub fn nextEpoch(self: EpochExecutor, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, generation: usize, population: *Population) !void {
        try switch (self) {
            inline else => |s| s.nextEpoch(allocator, rand, opts, generation, population),
        };
    }
};

/// Returns the appropriate executor type from given Options.
pub fn epochExecutorForCtx(allocator: std.mem.Allocator, ctx: *Options) !EpochExecutor {
    return switch (ctx.epoch_executor_type) {
        EpochExecutorType.EpochExecutorTypeSequential => {
            var executor = try SequentialPopulationEpochExecutor.init(allocator);
            return EpochExecutor{ .sequential = executor };
        },
        EpochExecutorType.EpochExecutorTypeParallel => {
            var executor = try ParallelPopulationEpochExecutor.init(allocator);
            return EpochExecutor{ .parallel = executor };
        },
    };
}

test {
    std.testing.refAllDecls(@This());
}
