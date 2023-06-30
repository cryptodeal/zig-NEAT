const std = @import("std");
const Network = @import("../network/network.zig").Network;
const neat_genome = @import("genome.zig");
const Species = @import("species.zig").Species;

const Genome = neat_genome.Genome;

pub const Organism = struct {
    // measure of fitness for the organism
    fitness: f64,
    // error value indicating how far organism's performance is from ideal task goal (e.g. MSE)
    error_value: f64 = undefined,
    // win marker (if needed for particular task)
    is_winner: bool = false,

    // organism's phenotype
    phenotype: ?*Network = null,
    // organism's genotype
    genotype: *Genome,
    // organism's species
    species: *Species = undefined,

    // number of children this organism may have
    expected_offspring: f64 = undefined,
    // indicates which generation this organism is from
    generation: usize,

    // utility data transfer object to be used by different GA implementations to hold
    // any addtl data; implemented as any to allow implementation specific objects
    // TODO: data: T = undefined,

    // fitness measure that won't change during fitness adjustments of population's epoch evaluation
    og_fitness: f64 = undefined,

    // marker for destruction of inferior organisms
    to_eliminate: bool = false,
    // marks the species champion
    is_champion: bool = false,

    // number of reserved offspring for a population leader
    super_champ_offspring: usize = undefined,
    // marks the best in population
    is_population_champion: bool = false,
    // marks the duplicate child of a champion (for tracking purposes)
    is_population_champion_child: bool = false,

    // DEBUG variable - highest fitness of champ
    highest_fitness: f64 = undefined,

    // track its origin - for debugging or analysis - specifies how organism was created
    mutation_struct_baby: bool = false,
    mate_baby: bool = false,

    // flag to be used as utility value
    flag: usize = undefined,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, fit: f64, g: *Genome, generation: usize) !*Organism {
        var phenotype: *Network = undefined;
        if (g.phenotype == null) {
            phenotype = try g.genesis(g.id);
        } else {
            phenotype = g.phenotype.?;
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

    pub fn deinit(self: *Organism) void {
        self.genotype.deinit();
        self.allocator.destroy(self);
    }

    pub fn update_phenotype(self: *Organism) !void {
        // TODO: might need to free phenotype
        self.phenotype = null;
        // recreate phenotype off new genotype
        self.phenotype = try self.genotype.genesis(self.genotype.id);
    }

    pub fn check_champion_child_damaged(self: *Organism) bool {
        if (self.is_population_champion_child and self.highest_fitness > self.fitness) {
            return true;
        }
        return false;
    }
};

pub fn fitness_comparison(context: void, a: *Organism, b: *Organism) bool {
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
    var gnome = try neat_genome.build_test_genome(allocator, 1);
    defer gnome.deinit();
    var count: usize = 100;
    var i: usize = 0;
    var orgs = try allocator.alloc(*Organism, count);
    defer allocator.free(orgs);
    while (i < count) : (i += 1) {
        var new_genome = try gnome.duplicate(@as(i64, @intCast(count)));
        orgs[i] = try Organism.init(allocator, rand.float(f64), new_genome, 1);
    }
    // sort ascending
    std.mem.sort(*Organism, orgs, {}, fitness_comparison);
    var fit: f64 = 0.0;
    for (orgs) |o| {
        try std.testing.expect(o.fitness > fit);
        fit = o.fitness;
        o.deinit();
    }

    // sort descending
    i = 0;
    while (i < count) : (i += 1) {
        var new_genome = try gnome.duplicate(@as(i64, @intCast(count)));
        orgs[i] = try Organism.init(allocator, rand.float(f64), new_genome, 1);
    }
    std.mem.sort(*Organism, orgs, {}, fitness_comparison);
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
    var gnome = try neat_genome.build_test_genome(allocator, 1);
    var org = try Organism.init(allocator, rand.float(f64), gnome, 1);
    defer org.deinit();

    org.is_population_champion_child = true;
    org.highest_fitness = 100;
    org.fitness = 1000;

    var res = org.check_champion_child_damaged();
    try std.testing.expect(!res);

    org.fitness = 10;
    res = org.check_champion_child_damaged();
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
    var gnome = try neat_genome.build_test_genome(allocator, 1);
    var org = try Organism.init(allocator, rand.float(f64), gnome, 1);
    defer org.deinit();

    org.phenotype = null;
    try std.testing.expect(org.phenotype == null);
    try org.update_phenotype();
    try std.testing.expect(org.phenotype != null);
}
