const std = @import("std");
const exp_trial = @import("trial.zig");
const neat_organism = @import("../genetics/organism.zig");
const maths = @import("floats.zig");
const neat_population = @import("../genetics/population.zig");
const Options = @import("../opts.zig").Options;
const exp_common = @import("common.zig");
const neat_genome = @import("../genetics/genome.zig");
const exp_generation = @import("generation.zig");

const Trial = exp_trial.Trial;
const Organism = neat_organism.Organism;
const Population = neat_population.Population;
const Genome = neat_genome.Genome;
const EpochExecutor = exp_common.EpochExecutor;
const GenerationEvaluator = exp_common.GenerationEvaluator;
const Generation = exp_generation.Generation;
const fitness_comparison = neat_organism.fitness_comparison;
const epoch_executor_for_ctx = exp_common.epoch_executor_for_ctx;

/// An Experiment is a collection of trials for one experiment.
/// It's helpful for analysis of a series of experiments.
pub const Experiment = struct {
    id: usize,
    name: []const u8 = undefined,
    rand_seed: u64 = undefined,
    trials: std.ArrayList(*Trial),
    /// The maximal allowed fitness score as defined by fitness function of experiment.
    /// It is used to normalize fitness score value used in efficiency score calculation. If this value
    /// is not set the fitness score will not be normalized during efficiency score estimation.
    max_fitness_score: ?f64 = null,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: usize) !*Experiment {
        var self = try allocator.create(Experiment);
        self.* = .{
            .allocator = allocator,
            .id = id,
            .trials = std.ArrayList(*Trial).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Experiment) void {
        for (self.trials.items) |t| {
            t.deinit();
        }
        self.trials.deinit();
        self.allocator.destroy(self);
    }

    /// `avg_trial_duration` Calculates average duration of experiment's trial. Returns EmptyDuration for experiment with no trials.
    /// Note, that most trials finish after solution solved, so this metric can be used to represent how efficient the solvers
    /// was generated
    pub fn avg_trial_duration(self: *Experiment) i64 {
        var total: u64 = 0;
        for (self.trials.items) |t| {
            total += t.duration;
        }
        if (self.trials.items.len > 0) {
            return @as(i64, @intCast(total / @as(u64, @intCast(self.trials.items.len))));
        } else {
            return -1;
        }
    }

    /// `avg_epoch_duration` Calculates average duration of evaluations among all generations of organism populations in this experiment
    pub fn avg_epoch_duration(self: *Experiment) i64 {
        var total: u64 = 0;
        for (self.trials.items) |t| {
            total += t.avg_epoch_duration();
        }
        if (self.trials.items.len > 0) {
            return @as(i64, @intCast(total / @as(u64, @intCast(self.trials.items.len))));
        } else {
            return -1;
        }
    }

    /// `avg_generation_per_trial` Calculates average number of generations evaluated per trial during this experiment.
    /// This can be helpful when estimating algorithm efficiency, because when winner organism is found the trial is
    /// terminated, i.e. less evaluations - more fast convergence.
    pub fn avg_generations_per_trial(self: *Experiment) f64 {
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

    /// `most_recent_trial_eval_time` Returns the time of evaluation of the most recent trial
    pub fn most_recent_trial_eval_time(self: *Experiment) ?std.time.Instant {
        if (self.trials.items.len == 0) {
            return null;
        }
        var u: std.time.Instant = undefined;
        for (self.trials.items, 0..) |t, i| {
            var ut = t.recent_epoch_eval_time();
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

    /// `best_organism` Finds the most fit organism among all trials in this experiment. It's also possible to get the best organism
    /// only among the ones which was able to solve the experiment's problem. Returns the best fit organism in this experiment
    /// among with ID of trial where it was found and boolean value to indicate if search was successful.
    pub fn best_organism(self: *Experiment, allocator: std.mem.Allocator, only_solvers: bool) !?*Organism {
        var orgs = std.ArrayList(*Organism).init(allocator);
        defer orgs.deinit();
        for (self.trials.items, 0..) |t, i| {
            var org = try t.best_organism(only_solvers);
            if (org != null) {
                try orgs.append(org);
                org.flag = i;
            }
        }
        if (orgs.items.len > 0) {
            std.mem.sort(*Organism, orgs.items, {}, fitness_comparison);
            std.mem.reverse(*Organism, orgs.items);
            return orgs.items[0];
        } else {
            return null;
        }
    }

    /// `solved` checks whether solution was found in at least one trial
    pub fn solved(self: *Experiment) bool {
        for (self.trials.items) |t| {
            if (t.solved()) {
                return true;
            }
        }
        return false;
    }

    /// `best_fitness` finds the fitness values of the best organisms for each trial.
    pub fn best_fitness(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            var org = try t.best_organism(false);
            if (org != null) {
                x[i] = org.fitness;
            }
        }
        return x;
    }

    /// `best_species_age` finds the age values of the species with the best organisms for each trial.
    pub fn best_species_age(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            var org = try t.best_organism(false);
            if (org != null) {
                x[i] = @as(f64, @floatFromInt(org.species.age));
            }
        }
        return x;
    }

    /// `best_complexity` finds the complexity values of the best organisms for each trial.
    pub fn best_complexity(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            var org = try t.best_organism(false);
            if (org != null) {
                x[i] = @as(f64, @floatFromInt(org.phenotype.complexity()));
            }
        }
        return x;
    }

    /// `avg_diversity` returns the average number of species in each trial
    pub fn avg_diversity(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            // TODO: arena allocator (w free and retain capacity, might be faster)
            var diversity = try t.diversity(allocator);
            defer allocator.free(diversity);
            x[i] = maths.mean(f64, diversity);
        }
        return x;
    }

    /// `epochs_per_trial` calculates the number of epochs per trial
    pub fn epochs_per_trial(self: *Experiment, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.trials.items.len);
        for (self.trials.items, 0..) |t, i| {
            x[i] = @as(f64, @floatFromInt(t.generations.items.len));
        }
        return x;
    }

    /// `trials_solved` calculates the number of trials solved
    pub fn trials_solved(self: *Experiment) u64 {
        var count: u64 = 0;
        for (self.trials.items) |t| {
            if (t.solved()) {
                count += 1;
            }
        }
        return count;
    }

    /// `success_rate` calculates the success rate of the experiment as ratio
    /// of trials with successful solvers per total number of trials executed.
    pub fn success_rate(self: *Experiment) f64 {
        var is_solved = @as(f64, @floatFromInt(self.trials_solved()));
        if (self.trials.items.len > 0) {
            return is_solved / @as(f64, @floatFromInt(self.trials.items.len));
        } else {
            return 0;
        }
    }

    /// `avg_winner_statistics` calculates the average number of nodes, genes, organisms evaluations,
    /// and species diversity of winners among all trials, i.e. for all trials where winning solution was found.
    pub fn avg_winner_statistics(self: *Experiment, allocator: std.mem.Allocator) !*AvgWinnerStats {
        var avg_winner_stats = try AvgWinnerStats.init(allocator);
        var count: f64 = 0;
        var total_nodes: i64 = 0;
        var total_genes: i64 = 0;
        var total_evals: i64 = 0;
        var total_diversity: i64 = 0;
        for (self.trials.items) |t| {
            if (t.solved()) {
                // TODO: arena allocator (w free and retain capacity, might be faster)
                var t_stats = try t.winner_statistics(allocator);
                defer t_stats.deinit();
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

    pub fn efficiency_score(self: *Experiment) f64 {
        var mean_complexity: f64 = 0;
        var mean_fitness: f64 = 0;
        if (self.trials.items.len > 0) {
            var count: f64 = 0;
            for (self.trials.items) |t| {
                if (t.solved()) {
                    if (t.winner_generation == null) {
                        // find winner
                        var stats = try t.winner_statistics();
                        defer stats.deinit();
                    }
                    mean_complexity += @as(f64, @floatFromInt(t.winner_generation.?.champion.phenotype.?.complexity()));
                    mean_fitness += t.winner_generation.?.champion.fitness;

                    count += 1;
                }
            }
            mean_complexity /= count;
            mean_fitness /= count;
        }

        // normalize and scale fitness score if appropriate
        var fitness_score = mean_fitness;
        if (self.max_fitness_score > 0) {
            fitness_score = (fitness_score / self.max_fitness_score) * 100;
        }

        var score = self.penalty_score(mean_complexity);
        if (score > 0) {
            // calculate normalized score
            var succeed_rate = self.success_rate();
            var log_penalty_score = @log(score);
            score = succeed_rate / log_penalty_score;
        }
        return score;
    }

    fn penalty_score(self: *Experiment, mean_complexity: f64) f64 {
        return @as(f64, @floatFromInt(self.avg_epoch_duration())) * self.avg_generations_per_trial() * mean_complexity;
    }

    pub fn execute(self: *Experiment, allocator: std.mem.Allocator, opts: *Options, start_genome: *Genome, comptime evaluator: GenerationEvaluator) !void {
        var run: usize = 0;
        while (run < opts.num_runs) : (run += 1) {
            var trial_start_time = try std.time.Instant.now();
            std.debug.print("\n>>>>> Spawning new population: ", .{});
            var pop = Population.init(allocator, start_genome, opts) catch |err| {
                std.debug.print("Failed to spawn new population from start genome\n", .{});
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
            var epoch_executor: EpochExecutor = try epoch_executor_for_ctx(allocator, opts);
            defer epoch_executor.deinit();

            // start new trial
            var trial = try Trial.init(self.allocator, run);
            errdefer trial.deinit();

            // TODO: implement/notify TrialObserver that run started

            var generation_id: usize = 0;
            while (generation_id < opts.num_generations) : (generation_id += 1) {
                std.debug.print("\n>>>>> Generation:{d}\tRun: {d}\n", .{ generation_id, run });
                var generation = try Generation.init(allocator, generation_id, run);
                var gen_start_time = std.time.Instant.now() catch unreachable;
                evaluator.generation_evaluate(opts, pop, generation) catch |err| {
                    std.debug.print("!!!!! Generation [{d}] evaluation failed !!!!!\n", .{generation_id});
                    generation.deinit_early();
                    return err;
                };

                generation.executed = std.time.Instant.now() catch unreachable;

                // Turnover population of organisms to the next epoch if appropriate
                if (!generation.solved) {
                    // std.debug.print(">>>>> start next generation\n", .{});
                    std.debug.print("\n\nNEXT EPOCH:\n\n", .{});
                    epoch_executor.next_epoch(opts, generation_id, pop) catch |err| {
                        std.debug.print("!!!!! Epoch execution failed in generation [{d}] !!!!!\n", .{generation_id});
                        return err;
                    };
                }

                // Set generation duration, which also includes preparation for the next epoch
                generation.duration = generation.executed.since(gen_start_time);
                try trial.generations.append(generation);

                // TODO: implement/notify TrialObserver

                if (generation.solved) {
                    // stop further evaluation if already solved
                    std.debug.print(">>>>> The winner organism found in [{d}] generation, fitness: {d} <<<<<\n", .{ generation_id, generation.champion.?.fitness });
                    // TODO: implement/notify TrialObserver
                    break;
                }
            }
            // holds trial duration
            var current_time = std.time.Instant.now() catch unreachable;
            trial.duration = current_time.since(trial_start_time);

            // store trial into experiment
            try self.trials.append(trial);

            // TODO: implement/notify TrialObserver
        }
    }
};

pub const AvgWinnerStats = struct {
    avg_nodes: f64 = -1,
    avg_genes: f64 = -1,
    avg_evals: f64 = -1,
    avg_diversity: f64 = -1,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*AvgWinnerStats {
        var self: *AvgWinnerStats = try allocator.create(AvgWinnerStats);
        self.* = .{
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *AvgWinnerStats) void {
        self.allocator.destroy(self);
    }
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

    var duration = exp.avg_trial_duration();
    try std.testing.expect(duration == 5);
}
