const std = @import("std");
const exp_trial = @import("trial.zig");
const neat_organism = @import("../genetics/organism.zig");
const maths = @import("floats.zig");
const neat_population = @import("../genetics/population.zig");
const neat_options = @import("../opts.zig");
const exp_common = @import("common.zig");
const neat_genome = @import("../genetics/genome.zig");
const exp_generation = @import("generation.zig");
const TrialRunObserver = @import("trial_run_observer.zig");

const Trial = exp_trial.Trial;
const Organism = neat_organism.Organism;
const Population = neat_population.Population;
const Genome = neat_genome.Genome;
const EpochExecutor = exp_common.EpochExecutor;
const Options = neat_options.Options;
const logger = @constCast(neat_options.logger);
const GenerationEvaluator = @import("generation_evaluator.zig");
const Generation = exp_generation.Generation;
const fitnessComparison = neat_organism.fitnessComparison;
const epochExecutorForCtx = exp_common.epochExecutorForCtx;

/// An Experiment is a collection of trials for one experiment.
/// It's helpful for analysis of a series of experiments.
pub const Experiment = struct {
    /// The Experiment's Id.
    id: usize,
    /// The Experiment's name.
    name: []const u8 = undefined,
    /// The list of all Trials comprising the Experiment.
    trials: std.ArrayList(*Trial),
    /// The maximal allowed fitness score as defined by fitness function of experiment.
    /// It is used to normalize fitness score value used in efficiency score calculation. If this value
    /// is not set the fitness score will not be normalized during efficiency score estimation.
    max_fitness_score: ?f64 = null,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new Experiment.
    pub fn init(allocator: std.mem.Allocator, id: usize) !*Experiment {
        var self = try allocator.create(Experiment);
        self.* = .{
            .allocator = allocator,
            .id = id,
            .trials = std.ArrayList(*Trial).init(allocator),
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *Experiment) void {
        for (self.trials.items) |t| t.deinit();
        self.trials.deinit();
        self.allocator.destroy(self);
    }

    /// Calculates average duration of the Experiment's Trials. Returns `-1` for Experiment with no trials.
    /// N.B. most Trials finish after solution has been found, so this metric can be used to represent how efficiently
    /// the solvers were generated
    pub fn avgTrialDuration(self: *Experiment) f64 {
        var total: u64 = 0;
        for (self.trials.items) |t| {
            total += t.duration;
        }
        if (self.trials.items.len > 0) {
            return @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(self.trials.items.len));
        } else {
            return -1;
        }
    }

    /// Calculates average duration of evaluations among all Generations of organism populations in this experiment.
    pub fn avgEpochDuration(self: *Experiment) f64 {
        var total: i64 = 0;
        for (self.trials.items) |t| {
            total += t.avgEpochDuration();
        }
        if (self.trials.items.len > 0) {
            return @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(self.trials.items.len));
        } else {
            return -1;
        }
    }

    /// Calculates average number of Generations evaluated per trial during this experiment.
    /// This can be helpful when estimating algorithm efficiency, because when winner organism is found the trial is
    /// terminated, i.e. less evaluations - more fast convergence.
    pub fn avgGenerationsPerTrial(self: *Experiment) f64 {
        var total: f64 = 0;
        for (self.trials.items) |t| {
            total += @as(f64, @floatFromInt(t.generations.items.len));
        }
        if (self.trials.items.len > 0) {
            return total / @as(f64, @floatFromInt(self.trials.items.len));
        } else {
            return 0;
        }
    }

    /// Returns the time of evaluation of the most recent trial.
    pub fn mostRecentTrialEvalTime(self: *Experiment) ?std.time.Instant {
        if (self.trials.items.len == 0) {
            return null;
        }
        var u: std.time.Instant = undefined;
        for (self.trials.items, 0..) |t, i| {
            var ut = t.recentEpochEvalTime();
            if (i == 0) {
                u = ut;
                continue;
            }
            if (u.order(ut) == .lt) {
                u = ut;
            }
        }
        return u;
    }

    /// Finds the most fit organism among all trials in this experiment. It's also possible to get the best organism
    /// only among the ones which was able to solve the experiment's problem. Returns the best fit organism in this experiment
    /// among with Id of trial where it was found and boolean value to indicate if search was successful.
    pub fn bestOrganism(self: *Experiment, allocator: std.mem.Allocator, only_solvers: bool) !?*Organism {
        var orgs = std.ArrayList(*Organism).init(allocator);
        defer orgs.deinit();
        for (self.trials.items, 0..) |t, i| {
            var org = try t.bestOrganism(allocator, only_solvers);
            if (org != null) {
                try orgs.append(org.?);
                org.?.flag = i;
            }
        }
        if (orgs.items.len > 0) {
            std.mem.sort(*Organism, orgs.items, {}, fitnessComparison);
            std.mem.reverse(*Organism, orgs.items);
            return orgs.items[0];
        } else {
            return null;
        }
    }

    /// Checks whether solution was found in at least one trial.
    pub fn solved(self: *Experiment) bool {
        for (self.trials.items) |t| {
            if (t.solved()) {
                return true;
            }
        }
        return false;
    }

    /// Finds the fitness values of the best organisms for each trial.
    pub fn bestFitness(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            var org = try t.bestOrganism(allocator, false);
            if (org != null) {
                x[i] = org.?.fitness;
            }
        }
        return x;
    }

    /// Finds the age values of the species with the best organisms for each trial.
    pub fn bestSpeciesAge(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            var org = try t.bestOrganism(allocator, false);
            if (org != null) {
                x[i] = @as(f64, @floatFromInt(org.?.species.age));
            }
        }
        return x;
    }

    /// Finds the complexity values of the best organisms for each trial.
    pub fn bestComplexity(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            var org = try t.bestOrganism(allocator, false);
            if (org != null) {
                x[i] = @as(f64, @floatFromInt(org.?.phenotype.?.complexity()));
            }
        }
        return x;
    }

    /// Returns the average number of species in each trial.
    pub fn avgDiversity(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            // TODO: arena allocator (w free and retain capacity, might be faster)
            var diversity = try t.diversity(allocator);
            defer allocator.free(diversity);
            x[i] = maths.mean(f64, diversity);
        }
        return x;
    }

    /// Calculates the number of epochs per trial.
    pub fn epochsPerTrial(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            x[i] = @as(f64, @floatFromInt(t.generations.items.len));
        }
        return x;
    }

    /// Calculates the number of trials solved.
    pub fn trialsSolved(self: *Experiment) u64 {
        var count: u64 = 0;
        for (self.trials.items) |t| {
            if (t.solved()) {
                count += 1;
            }
        }
        return count;
    }

    /// Calculates the success rate of the experiment as ratio
    /// of trials with successful solvers per total number of trials executed.
    pub fn successRate(self: *Experiment) f64 {
        var is_solved = @as(f64, @floatFromInt(self.trialsSolved()));
        if (self.trials.items.len > 0) {
            return is_solved / @as(f64, @floatFromInt(self.trials.items.len));
        } else {
            return 0;
        }
    }

    /// Calculates the average number of nodes, genes, organisms evaluations,
    /// and species diversity of winners among all trials, i.e. for all trials where winning solution was found.
    pub fn avgWinnerStats(self: *Experiment) *AvgWinnerStats {
        var avg_winner_stats = AvgWinnerStats{};
        var count: f64 = 0;
        var total_nodes: i64 = 0;
        var total_genes: i64 = 0;
        var total_evals: i64 = 0;
        var total_diversity: i64 = 0;
        for (self.trials.items) |t| {
            if (t.solved()) {
                // TODO: arena allocator (w free and retain capacity, might be faster)
                var t_stats = t.winnerStats();
                total_nodes += t_stats.nodes;
                total_genes += t_stats.genes;
                total_evals += t_stats.evals;
                total_diversity += t_stats.diversity;

                count += 1;
            }
        }
        if (count > 0) {
            avg_winner_stats.avg_nodes = @as(f64, @floatFromInt(total_nodes)) / count;
            avg_winner_stats.avg_genes = @as(f64, @floatFromInt(total_genes)) / count;
            avg_winner_stats.avg_evals = @as(f64, @floatFromInt(total_evals)) / count;
            avg_winner_stats.avg_diversity = @as(f64, @floatFromInt(total_diversity)) / count;
        }
        return avg_winner_stats;
    }

    /// Calculates the efficiency score of the found solution.
    ///
    /// We are interested in efficient solver solution that take less time per epoch, fewer generations per trial,
    /// and produce less complicated winner genomes. At the same time it should have maximal fitness score and maximal
    /// success rate among trials.
    ///
    /// This value can only be compared against values obtained for the same type of experiments.
    pub fn efficiencyScore(self: *Experiment) f64 {
        var mean_complexity: f64 = 0;
        var mean_fitness: f64 = 0;
        if (self.trials.items.len > 0) {
            var count: f64 = 0;
            for (self.trials.items) |t| {
                if (t.solved()) {
                    if (t.winner_generation == null) {
                        // find winner
                        _ = t.winnerStats();
                    }
                    mean_complexity += @as(f64, @floatFromInt(t.winner_generation.?.champion.?.phenotype.?.complexity()));
                    mean_fitness += t.winner_generation.?.champion.?.fitness;

                    count += 1;
                }
            }
            mean_complexity /= count;
            mean_fitness /= count;
        }

        // normalize and scale fitness score if appropriate
        var fitness_score = mean_fitness;
        if (self.max_fitness_score != null and self.max_fitness_score.? > 0) {
            fitness_score = (fitness_score / self.max_fitness_score.?) * 100;
        }

        var score = self.penaltyScore(mean_complexity);
        if (score > 0) {
            // calculate normalized score
            var succeed_rate = self.successRate();
            var log_penalty_score = @log(score);
            score = succeed_rate / log_penalty_score;
        }
        return score;
    }

    fn penaltyScore(self: *Experiment, mean_complexity: f64) f64 {
        return self.avgEpochDuration() * self.avgGenerationsPerTrial() * mean_complexity;
    }

    /// Used to run specific experiment using provided `start_genome` and specific evaluator for each epoch of the experiment.
    pub fn execute(self: *Experiment, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, start_genome: *Genome, evaluator: GenerationEvaluator, trial_observer: ?TrialRunObserver) !void {
        var run: usize = 0;
        while (run < opts.num_runs) : (run += 1) {
            var trial_start_time = try std.time.Instant.now();
            logger.info(">>>>> Spawning new population: ", .{}, @src());
            var pop = Population.init(allocator, rand, start_genome, opts) catch |err| {
                logger.info("Failed to spawn new population from start genome\n", .{}, @src());
                return err;
            };
            defer pop.deinit();
            std.debug.print("OK <<<<<\n>>>>> Verifying spawned population:", .{});
            _ = pop.verify() catch |err| {
                std.debug.print("\n!!!!! Population verification failed !!!!!", .{});
                return err;
            };
            std.debug.print("OK <<<<<", .{});

            // create appropriate population's epoch executor
            var epoch_executor: EpochExecutor = try epochExecutorForCtx(allocator, opts);
            defer epoch_executor.deinit();

            // start new trial
            var trial = try Trial.init(allocator, run);

            if (trial_observer != null) {
                trial_observer.?.trialRunStarted(trial);
            }

            var generation_id: usize = 0;
            while (generation_id < opts.num_generations) : (generation_id += 1) {
                std.debug.print("\n>>>>> Generation:{d}\tRun: {d}\n", .{ generation_id, run });
                var generation = try Generation.init(allocator, generation_id, run);
                var gen_start_time = std.time.Instant.now() catch unreachable;
                evaluator.generationEvaluate(opts, pop, generation) catch |err| {
                    std.debug.print("!!!!! Generation [{d}] evaluation failed !!!!!\n", .{generation_id});
                    generation.deinitEarly();
                    return err;
                };

                generation.executed = std.time.Instant.now() catch unreachable;

                // Turnover population of organisms to the next epoch if appropriate
                if (!generation.solved) {
                    // std.debug.print(">>>>> start next generation\n", .{});
                    std.debug.print("\n\nNEXT EPOCH:\n\n", .{});
                    epoch_executor.nextEpoch(allocator, rand, opts, generation_id, pop) catch |err| {
                        std.debug.print("!!!!! Epoch execution failed in generation [{d}] !!!!!\n", .{generation_id});
                        return err;
                    };
                }

                // Set generation duration, which also includes preparation for the next epoch
                generation.duration = generation.executed.since(gen_start_time);
                try trial.generations.append(generation);

                if (trial_observer != null) {
                    trial_observer.?.epochEvaluated(trial, generation);
                }

                if (generation.solved) {
                    // stop further evaluation if already solved
                    std.debug.print(">>>>> The winner organism found in [{d}] generation, fitness: {d} <<<<<\n", .{ generation_id, generation.champion.?.fitness });
                    // TODO: implement/notify TrialObserver
                    break;
                }
            }
            // holds trial duration
            var current_time = try std.time.Instant.now();
            trial.duration = current_time.since(trial_start_time);

            // store trial into experiment
            self.trials.appendAssumeCapacity(trial);

            if (trial_observer != null) {
                trial_observer.?.trialRunFinished(trial);
            }
        }
    }
};

pub const AvgWinnerStats = struct {
    avg_nodes: f64 = -1,
    avg_genes: f64 = -1,
    avg_evals: f64 = -1,
    avg_diversity: f64 = -1,
};

test "Experiment avg Trial duration" {
    var allocator = std.testing.allocator;
    var exp = try Experiment.init(allocator, 1);
    defer exp.deinit();
    var trial1 = try Trial.init(allocator, 1);
    trial1.duration = 3;
    try exp.trials.append(trial1);
    var trial2 = try Trial.init(allocator, 2);
    trial2.duration = 10;
    try exp.trials.append(trial2);
    var trial3 = try Trial.init(allocator, 3);
    trial3.duration = 2;
    try exp.trials.append(trial3);

    var duration = exp.avgTrialDuration();
    try std.testing.expect(duration == 5);
}
