const std = @import("std");
const pop_epoch = @import("../genetics/population_epoch.zig");
const neat_options = @import("../opts.zig");
const neat_population = @import("../genetics/population.zig");
const exp_generation = @import("generation.zig");
const experiment = @import("experiment.zig");
const trial = @import("trial.zig");

// exports
pub const Experiment = experiment.Experiment;
pub const AvgWinnerStats = experiment.AvgWinnerStats;
pub const floats = @import("floats.zig");
pub const Generation = exp_generation.Generation;
pub const GenerationAvg = exp_generation.GenerationAvg;
pub const Trial = trial.Trial;
pub const WinnerStats = trial.WinnerStats;
pub const TrialAvg = trial.TrialAvg;

const SequentialPopulationEpochExecutor = pop_epoch.SequentialPopulationEpochExecutor;
const ParallelPopulationEpochExecutor = pop_epoch.ParallelPopulationEpochExecutor;
const Options = neat_options.Options;
const EpochExecutorType = neat_options.EpochExecutorType;
const Population = neat_population.Population;

pub const EpochExecutor = union(enum) {
    sequential: *SequentialPopulationEpochExecutor,
    parallel: *ParallelPopulationEpochExecutor,

    pub fn deinit(self: EpochExecutor) void {
        switch (self) {
            inline else => |s| s.deinit(),
        }
    }

    pub fn next_epoch(self: EpochExecutor, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, generation: usize, population: *Population) !void {
        try switch (self) {
            inline else => |s| s.next_epoch(allocator, rand, opts, generation, population),
        };
    }
};

pub fn epoch_executor_for_ctx(allocator: std.mem.Allocator, ctx: *Options) !EpochExecutor {
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

pub const GenerationEvaluator = struct {
    generation_evaluate: *const fn (*Options, *Population, *Generation, *anyopaque) anyerror!void,
    // TODO: use ctx field to store arbitrary data for the evaluator
    ctx: *anyopaque = undefined,
};

test {
    std.testing.refAllDecls(@This());
}
