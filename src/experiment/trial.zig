const std = @import("std");
const Generation = @import("generation.zig").Generation;
const neat_organism = @import("../genetics/organism.zig");

const Organism = neat_organism.Organism;
const fitness_comparison = neat_organism.fitness_comparison;

pub const Trial = struct {
    // The trial number
    id: u64,
    // The results per generation in this trial
    generations: std.ArrayList(*Generation),
    // The winner generation
    winner_generation: ?*Generation = null,

    // The elapsed time between trial start and finish
    duration: u64 = undefined,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: u64) !*Trial {
        var self = try allocator.create(Trial);
        self.* = .{
            .id = id,
            .allocator = allocator,
            .generations = std.ArrayList(*Generation).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Trial) void {
        for (self.generations.items) |gen| {
            gen.deinit();
        }
        self.generations.deinit();
        self.allocator.destroy(self);
    }

    pub fn avg_epoch_duration(self: *Trial) i64 {
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

    pub fn recent_epoch_eval_time(self: *Trial) std.time.Instant {
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

    pub fn best_organism(self: *Trial, only_solvers: bool) !?*Organism {
        var orgs = std.ArrayList(*Organism).init(self.allocator);
        defer orgs.deinit();
        for (self.generations.items) |e| {
            if (!only_solvers) {
                try orgs.append(e.champion.?);
            } else if (e.solved) {
                try orgs.append(e.champion.?);
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

    pub fn solved(self: *Trial) bool {
        for (self.generations.items) |e| {
            if (e.solved) {
                return true;
            }
        }
        return false;
    }

    pub fn champions_fitness(self: *Trial, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            x[i] = e.champion.?.fitness;
        }
        return x;
    }

    pub fn champion_species_ages(self: *Trial, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            x[i] = @as(f64, @floatFromInt(e.champion.?.species.age));
        }
        return x;
    }

    pub fn champions_complexities(self: *Trial, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            x[i] = @as(f64, @floatFromInt(e.champion.?.phenotype.?.complexity()));
        }
        return x;
    }

    pub fn diversity(self: *Trial, allocator: std.mem.Allocator) ![]f64 {
        var x = try allocator.alloc(f64, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            x[i] = @as(f64, @floatFromInt(e.diversity));
        }
        return x;
    }

    pub fn average(self: *Trial, allocator: std.mem.Allocator) !*TrialAvg {
        var self_avg = try TrialAvg.init(allocator, self.generations.items.len);
        for (self.generations.items, 0..) |e, i| {
            var gen_avg = try e.average(allocator);
            defer gen_avg.deinit();
            self_avg.fitness[i] = gen_avg.fitness;
            self_avg.age[i] = gen_avg.age;
            self_avg.complexity[i] = gen_avg.complexity;
        }
        return self_avg;
    }

    pub fn winner_statistics(self: *Trial, allocator: std.mem.Allocator) !*WinnerStats {
        var stats = try WinnerStats.init(allocator);
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
        return stats;
    }
};

pub const WinnerStats = struct {
    nodes: i64 = -1,
    genes: i64 = -1,
    evals: i64 = -1,
    diversity: i64 = -1,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*WinnerStats {
        var self = try allocator.create(WinnerStats);
        self.* = .{
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *WinnerStats) void {
        self.allocator.destroy(self);
    }
};

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
