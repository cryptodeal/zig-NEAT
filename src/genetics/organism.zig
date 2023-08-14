const std = @import("std");
const Network = @import("../network/network.zig").Network;
const neat_genome = @import("genome.zig");
const Species = @import("species.zig").Species;

const Genome = neat_genome.Genome;

/// Organism is Genotypes (Genomes) and Phenotypes (Networks) with fitness information,
/// i.e. the genotype and phenotype together.
pub const Organism = struct {
    /// Measure of fitness for the organism.
    fitness: f64,
    /// The error value indicating how far organism's performance is from ideal task goal (e.g. MSE).
    error_value: f64 = undefined,
    /// Win marker (if needed for particular task).
    is_winner: bool = false,

    /// The organism's phenotype (Network).
    phenotype: ?*Network = null,
    // The organism's genotype (Genome).
    genotype: *Genome,
    // The organism's Species.
    species: *Species = undefined,

    /// The number of offspring this organism may have.
    expected_offspring: f64 = 0,
    /// Indicates which generation this organism is from.
    generation: usize,

    /// Utility data transfer object to be used by different GA implementations to hold
    /// any additional data; implemented as any to allow implementation specific objects.
    data: ?*anyopaque = null,

    /// Fitness measure that won't change during fitness adjustments of Population's epoch evaluation.
    og_fitness: f64 = undefined,

    /// Marker denoting whether organism should be eliminated while removing inferior organisms
    to_eliminate: bool = false,
    /// Marks the Species' champion.
    is_champion: bool = false,

    /// The number of reserved offspring for a Population leader.
    super_champ_offspring: usize = 0,
    /// Marks whether organism was the best in its Population.
    is_population_champion: bool = false,
    /// Marks the duplicate child of a champion (for tracking purposes).
    is_population_champion_child: bool = false,

    /// DEBUG variable - highest fitness of champ.
    highest_fitness: f64 = undefined,

    // track its origin - for debugging or analysis - specifies how the organism was created.

    /// Indicates whether organism was created via mutation (for debugging/analysis).
    mutation_struct_baby: bool = false,
    /// Indicates whether organism was created via mating (for debugging/analysis).
    mate_baby: bool = false,

    /// Flag to be used as utility value.
    flag: usize = undefined,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new Organism.
    pub fn init(allocator: std.mem.Allocator, fit: f64, g: *Genome, generation: usize) !*Organism {
        var phenotype: ?*Network = g.phenotype;
        if (phenotype == null) {
            phenotype = try g.genesis(allocator, g.id);
        }
        var res = try allocator.create(Organism);
        res.* = .{
            .allocator = allocator,
            .fitness = fit,
            .genotype = g,
            .phenotype = phenotype,
            .generation = generation,
        };
        return res;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *Organism) void {
        self.genotype.deinit();
        self.allocator.destroy(self);
    }

    /// Regenerates the underlying network graph based on a change in the genotype.
    pub fn updatePhenotype(self: *Organism, allocator: std.mem.Allocator) !void {
        if (self.phenotype != null) {
            self.phenotype.?.deinit();
            self.phenotype = null;
        }
        // recreate phenotype off new genotype
        self.phenotype = try self.genotype.genesis(allocator, self.genotype.id);
    }

    /// Checks if this organism is a child of the champion, but has the fitness score less than of the parent.
    /// This can be used to check if champion's offsprings degraded.
    pub fn checkChampionChildDamaged(self: *Organism) bool {
        if (self.is_population_champion_child and self.highest_fitness > self.fitness) {
            return true;
        }
        return false;
    }

    /// Formats Organism for printing to writer.
    pub fn format(value: Organism, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("[Organism generation: {d}, fitness: {d:.3}, original fitness: {d:.3}", .{ value.generation, value.fitness, value.og_fitness });
        if (value.is_champion) {
            try writer.writeAll(" - CHAMPION - ");
        }
        if (value.to_eliminate) {
            try writer.writeAll(" - TO BE ELIMINATED - ");
        }
        try writer.writeByte(']');
    }
};

/// Compares the fitness of two organisms; used to sort Slice of Organisms by fitness.
/// If fitness is equal, then the complexity of the phenotype is compared.
pub fn fitnessComparison(context: void, a: *Organism, b: *Organism) bool {
    _ = context;
    const org1 = a.*;
    const org2 = b.*;
    if (org1.fitness < org2.fitness) {
        return true;
    } else if (org1.fitness == org2.fitness) {
        // try to promote less complex organism
        var complexity1 = org1.phenotype.?.complexity();
        var complexity2 = org2.phenotype.?.complexity();
        if (complexity1 > complexity2) {
            return true; // higher complexity is less
        } else if (complexity1 == complexity2) {
            return org1.genotype.id < org2.genotype.id;
        }
    }
    return false;
}

test "Organisms sorting" {
    var allocator = std.testing.allocator;
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    var gnome = try neat_genome.buildTestGenome(allocator, 1);
    defer gnome.deinit();
    var count: usize = 100;
    var i: usize = 0;
    var orgs = try allocator.alloc(*Organism, count);
    defer allocator.free(orgs);
    while (i < count) : (i += 1) {
        var new_genome = try gnome.duplicate(allocator, @as(i64, @intCast(count)));
        orgs[i] = try Organism.init(allocator, rand.float(f64), new_genome, 1);
    }
    // sort ascending
    std.mem.sort(*Organism, orgs, {}, fitnessComparison);
    var fit: f64 = 0.0;
    for (orgs) |o| {
        try std.testing.expect(o.fitness > fit);
        fit = o.fitness;
        o.deinit();
    }

    // sort descending
    i = 0;
    while (i < count) : (i += 1) {
        var new_genome = try gnome.duplicate(allocator, @as(i64, @intCast(count)));
        orgs[i] = try Organism.init(allocator, rand.float(f64), new_genome, 1);
    }
    std.mem.sort(*Organism, orgs, {}, fitnessComparison);
    std.mem.reverse(*Organism, orgs);
    fit = std.math.inf(f64);
    for (orgs) |o| {
        try std.testing.expect(o.fitness < fit);
        fit = o.fitness;
        o.deinit();
    }
}

test "Organism champion child damaged" {
    var allocator = std.testing.allocator;
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    var gnome = try neat_genome.buildTestGenome(allocator, 1);
    var org = try Organism.init(allocator, rand.float(f64), gnome, 1);
    defer org.deinit();

    org.is_population_champion_child = true;
    org.highest_fitness = 100;
    org.fitness = 1000;

    var res = org.checkChampionChildDamaged();
    try std.testing.expect(!res);

    org.fitness = 10;
    res = org.checkChampionChildDamaged();
    try std.testing.expect(res);
}

test "Organism update phenotype" {
    var allocator = std.testing.allocator;
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    var gnome = try neat_genome.buildTestGenome(allocator, 1);
    var org = try Organism.init(allocator, rand.float(f64), gnome, 1);
    defer org.deinit();

    org.phenotype = null;
    try std.testing.expect(org.phenotype == null);
    try org.updatePhenotype(allocator);
    try std.testing.expect(org.phenotype != null);
}
