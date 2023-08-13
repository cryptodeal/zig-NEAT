const std = @import("std");
const Generation = @import("generation.zig").Generation;
const neat_organism = @import("../genetics/organism.zig");

const Organism = neat_organism.Organism;
const fitnessComparison = neat_organism.fitnessComparison;

// Trial holds statistics about one experiment run.
pub const Trial = struct {
    /// The trial number.
    id: u64,
    /// The results per generation in this trial.
    generations: std.ArrayList(*Generation),
    /// The winner generation.
    winner_generation: ?*Generation = null,

    /// The elapsed time between trial start and finish.
    duration: u64 = undefined,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new Trial.
    pub fn init(allocator: std.mem.Allocator, id: u64) !*Trial {
        var self = try allocator.create(Trial);
        self.* = .{
            .id = id,
            .allocator = allocator,
            .generations = std.ArrayList(*Generation).init(allocator),
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *Trial) void {
        for (self.generations.items) |gen| {
            gen.deinit();
        }
        self.generations.deinit();
        self.allocator.destroy(self);
    }

    /// Calculates average duration of evaluations among all
    /// generations of organism populations in this trial.
    pub fn avgEpochDuration(self: *Trial) i64 {
        var total: u64 = 0;
        for (self.generations.items) |generation| {
            total += generation.duration;
        }
        if (self.generations.items.len > 0) {
            return @as(i64, @intCast(total / @as(u64, @intCast(self.generations.items.len))));
        } else {
            return -1;
        }
    }

    /// Used to get time of the epoch executed most recently within this trial.
    pub fn recentEpochEvalTime(self: *Trial) std.time.Instant {
        var u: std.time.Instant = undefined;
        for (self.generations.items, 0..) |i, idx| {
            if (idx == 0) {
                u = i.executed;
                continue;
            }
            if (u.order(i.executed) == .lt) {
                u = i.executed;
            }
        }
        return u;
    }

    /// Finds the most fit organism among all epochs in this trial. It's also possible to
    /// get the best organism only among successful solvers of the experiment's problem.
    pub fn bestOrganism(self: *Trial, allocator: std.mem.Allocator, only_solvers: bool) !?*Organism {
        var orgs = std.ArrayList(*Organism).init(allocator);
        defer orgs.deinit();
        for (self.generations.items) |e| {
            if (!only_solvers) {
                try orgs.append(e.champion.?);
            } else if (e.solved) {
                try orgs.append(e.champion.?);
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

    /// Check whether any of the generations in this trial has solved the problem.
    pub fn solved(self: *Trial) bool {
        for (self.generations.items) |e| {
            if (e.solved) {
                return true;
            }
        }
        return false;
    }

    /// Returns the fitness values of the champion organisms per generation in this trial.
    pub fn championsFitness(self: *Trial, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            x[i] = e.champion.?.fitness;
        }
        return x;
    }

    /// Returns the age of the species of the champion per generation in this trial.
    pub fn championSpeciesAges(self: *Trial, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            x[i] = @as(f64, @floatFromInt(e.champion.?.species.age));
        }
        return x;
    }

    /// Returns the complexities of the champion organisms per generation in this trial.
    pub fn championsComplexities(self: *Trial, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            x[i] = @as(f64, @floatFromInt(e.champion.?.phenotype.?.complexity()));
        }
        return x;
    }

    /// Returns number of species for each epoch.
    pub fn diversity(self: *Trial, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            x[i] = @as(f64, @floatFromInt(e.diversity));
        }
        return x;
    }

    /// Average the average fitness, age, and complexity of the best organisms per species for each epoch in this trial.
    pub fn average(self: *Trial, allocator: std.mem.Allocator) *TrialAvg {
        var self_avg = try TrialAvg.init(allocator, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            var gen_avg = e.average();
            self_avg.fitness[i] = gen_avg.fitness;
            self_avg.age[i] = gen_avg.age;
            self_avg.complexity[i] = gen_avg.complexity;
        }
        return self_avg;
    }

    /// Finds the number of nodes, genes of the winner genome as well as number of winner
    /// organism evaluations and species diversity in the population with successful solver.
    pub fn winnerStats(self: *Trial) WinnerStats {
        var stats = WinnerStats{};
        if (self.winner_generation != null) {
            stats.nodes = @as(i64, @intCast(self.winner_generation.?.winner_nodes));
            stats.genes = @as(i64, @intCast(self.winner_generation.?.winner_genes));
            stats.evals = @as(i64, @intCast(self.winner_generation.?.winner_evals));
            stats.diversity = @as(i64, @intCast(self.winner_generation.?.diversity));
        } else if (self.generations.items.len > 0) {
            for (self.generations.items) |e| {
                if (e.solved) {
                    stats.nodes = @as(i64, @intCast(e.winner_nodes));
                    stats.genes = @as(i64, @intCast(e.winner_genes));
                    stats.evals = @as(i64, @intCast(e.winner_evals));
                    stats.diversity = @as(i64, @intCast(e.diversity));
                    // store winner
                    self.winner_generation = e;
                    break;
                }
            }
        }
        return &stats;
    }
};

/// Holds statistics about the winner genome of a trial.
pub const WinnerStats = struct {
    nodes: i64 = -1,
    genes: i64 = -1,
    evals: i64 = -1,
    diversity: i64 = -1,
};

/// Holds the average fitness, age, and complexity of the best organisms per species for each epoch in a given.
pub const TrialAvg = struct {
    fitness: []f64,
    complexity: []f64,
    age: []f64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, count: usize) !*TrialAvg {
        var self = try allocator.create(TrialAvg);
        self.* = .{
            .allocator = allocator,
            .fitness = try allocator.alloc(f64, count),
            .complexity = try allocator.alloc(f64, count),
            .age = try allocator.alloc(f64, count),
        };
        return self;
    }

    pub fn deinit(self: *TrialAvg) void {
        self.allocator.free(self.fitness);
        self.allocator.free(self.complexity);
        self.allocator.free(self.age);
        self.allocator.destroy(self);
    }
};
