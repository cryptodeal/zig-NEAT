const std = @import("std");
const pop_epoch = @import("../genetics/population_epoch.zig");
const neat_options = @import("../opts.zig");
const neat_population = @import("../genetics/population.zig");
const exp_generation = @import("generation.zig");
const experiment = @import("experiment.zig");
const trial = @import("trial.zig");
const utils = @import("utils.zig");

// exports

/// An Experiment is a collection of trials for one experiment.
pub const Experiment = experiment.Experiment;
/// Holds the average number of nodes, genes, organisms evaluations,
/// and species diversity of winners among all trials in the experiment.
pub const AvgWinnerStats = experiment.AvgWinnerStats;
/// Utility functions for statistical analysis of floating point slices.
pub const floats = @import("floats.zig");
/// Data structure representing the execution results of one generation.
pub const Generation = exp_generation.Generation;
/// Data structure that holds the average statistics of one `Generation`.
pub const GenerationAvg = exp_generation.GenerationAvg;
/// Data structure that holds statistics for one experiment run.
pub const Trial = trial.Trial;
/// Holds statistics about the winner genome of a trial.
pub const WinnerStats = trial.WinnerStats;
/// Holds the average fitness, age, and complexity of the best organisms per species for each epoch in a given.
pub const TrialAvg = trial.TrialAvg;
/// Utility function that creates the output directory when writing the results of a given trial.
pub const createOutDirForTrial = utils.createOutDirForTrial;
/// The standard TrialRunObserver interface.
pub const TrialRunObserver = @import("trial_run_observer.zig");
/// The standard GenerationEvaluator interface.
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
