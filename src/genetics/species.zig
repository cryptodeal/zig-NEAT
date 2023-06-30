const std = @import("std");
const orgn = @import("organism.zig");
const neat_population = @import("population.zig");
const neat_common = @import("common.zig");
const neat_genome = @import("genome.zig");

const Organism = orgn.Organism;
const Genome = neat_genome.Genome;
const Options = @import("../opts.zig").Options;
const Population = neat_population.Population;
const MutatorType = neat_common.MutatorType;
const fitness_comparison = orgn.fitness_comparison;

pub const MaxAvgFitness = struct {
    max: f64,
    avg: f64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*MaxAvgFitness {
        var res = try allocator.create(MaxAvgFitness);
        res.* = .{
            .allocator = allocator,
            .max = 0.0,
            .avg = 0.0,
        };
        return res;
    }

    pub fn deinit(self: *MaxAvgFitness) void {
        self.allocator.destroy(self);
    }
};

pub const OffspringCount = struct {
    expected: i64,
    skim: f64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*OffspringCount {
        var res = try allocator.create(OffspringCount);
        res.* = .{
            .allocator = allocator,
            .expected = 0,
            .skim = 0.0,
        };
        return res;
    }

    pub fn deinit(self: *OffspringCount) void {
        self.allocator.destroy(self);
    }
};

pub const Species = struct {
    // species ID
    id: i64,
    // age of the species
    age: i64 = 1,
    // maximal fitness of species (all time)
    max_fitness_ever: f64 = undefined,
    // how many offspring expected
    expected_offspring: i64 = undefined,

    // is it novel
    is_novel: bool = false,

    // organisms in the species; the algo keeps it sorted to have most
    // fit first at beginning of each reproduction cycle
    organisms: std.ArrayList(*Organism),
    // if this is too long ago, species will go extinct
    age_of_last_improvement: i64 = undefined,

    // flag used for search optimization
    is_checked: bool = false,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: i64) !*Species {
        var self = try allocator.create(Species);
        self.* = .{
            .allocator = allocator,
            .id = id,
            .organisms = std.ArrayList(*Organism).init(allocator),
        };
        return self;
    }

    pub fn init_novel(allocator: std.mem.Allocator, id: i64, novel: bool) !*Species {
        var self = try Species.init(allocator, id);
        self.is_novel = novel;
        return self;
    }

    pub fn deinit(self: *Species) void {
        for (self.organisms.items) |o| {
            o.deinit();
        }
        self.organisms.deinit();
        self.allocator.destroy(self);
    }

    pub fn add_organism(self: *Species, o: *Organism) !void {
        try self.organisms.append(o);
    }

    pub fn remove_organism(self: *Species, org: *Organism) !void {
        var old_args = self.organisms;
        var orgs = std.ArrayList(*Organism).init(self.allocator);

        for (old_args.items) |o| {
            if (!std.meta.eql(o.*, org.*)) {
                try orgs.append(o);
            } else {
                o.deinit();
            }
        }

        if (orgs.items.len != old_args.items.len) {
            std.debug.print("attempt to remove nonexistent Organism from Species with #of organisms: {d}\n", .{old_args.items.len});
            return error.FailedToRemoveOrganism;
        } else {
            old_args.deinit();
            self.organisms = orgs;
        }
    }

    pub fn adjust_fitness(self: *Species, opts: *Options) void {
        var age_debt = (self.age - self.age_of_last_improvement + 1) - opts.dropoff_age;
        if (age_debt == 0) {
            age_debt = 1;
        }

        for (self.organisms.items) |o| {
            // remember the original fitness before modifying
            o.og_fitness = o.fitness;

            // make fitness decrease after a stagnation point dropoff_age
            // added as if to keep species pristine until the dropoff point
            if (age_debt >= 1) {
                // extreme penalty for a long period of stagnation (divide fitness by 100)
                o.fitness = o.fitness * 0.01;
            }

            // give fitness boost to some young age (niching)
            // the age_significance parameter is a system parameter
            // if it is 1, then young species get no fitness boost
            if (self.age <= 10) {
                o.fitness = o.fitness * opts.age_significance;
            }
            // do not allow negative fitness
            if (o.fitness < 0.0) {
                o.fitness = 0.0001;
            }

            // share fitness w the species
            o.fitness = o.fitness / @as(f64, @floatFromInt(self.organisms.items.len));
        }

        // sort the population (most fit first) and mark for death those after : survival_threshold * pop_size
        std.mem.sort(*Organism, self.organisms.items, {}, fitness_comparison);
        std.mem.reverse(*Organism, self.organisms.items);

        // update age of last improvement
        if (self.organisms.items[0].og_fitness > self.max_fitness_ever) {
            self.age_of_last_improvement = self.age;
            self.max_fitness_ever = self.organisms.items[0].og_fitness;
        }

        // decide how many get to reproduce based on survival_thresh * pop_size
        // adding 1.0 ensures that at least one will survive
        var num_parents = @as(usize, @intFromFloat(@floor(opts.survival_thresh * @as(f64, @floatFromInt(self.organisms.items.len)) + 1.0)));

        // mark for death those who are ranked too low to reproduce
        self.organisms.items[0].is_champion = true; // flag champion
        while (num_parents < self.organisms.items.len) : (num_parents += 1) {
            self.organisms.items[num_parents].to_eliminate = true;
        }
    }

    pub fn compute_max_and_avg_fitness(self: *Species) !*MaxAvgFitness {
        var res = try MaxAvgFitness.init(self.allocator);
        var total: f64 = 0.0;
        for (self.organisms.items) |o| {
            total += o.fitness;
            if (o.fitness > res.max) {
                res.max = o.fitness;
            }
        }
        if (self.organisms.items.len > 0) {
            res.avg = total / @as(f64, @floatFromInt(self.organisms.items.len));
        }
        return res;
    }

    pub fn find_champion(self: *Species) ?*Organism {
        var champ_fitness: f64 = 0.0;
        var champ: ?*Organism = null;

        for (self.organisms.items) |org| {
            if (org.fitness > champ_fitness) {
                champ_fitness = org.fitness;
                champ = org;
            }
        }
        return champ;
    }

    pub fn first_organism(self: *Species) ?*Organism {
        if (self.organisms.items.len > 0) {
            return self.organisms.items[0];
        } else {
            return null;
        }
    }

    pub fn count_offspring(self: *Species, skim: f64) !*OffspringCount {
        var org_off_int_part: i64 = undefined;
        var org_off_fract_part: f64 = undefined;
        var skim_int_part: f64 = undefined;

        var res = try OffspringCount.init(self.allocator);
        res.expected = 0;
        res.skim = skim;

        for (
            self.organisms.items,
        ) |o| {
            org_off_int_part = @as(i64, @intFromFloat(@floor(o.expected_offspring)));
            org_off_fract_part = try std.math.mod(f64, o.expected_offspring, 1.0);

            res.expected += org_off_int_part;

            // skim off the fractional offspring
            res.skim += org_off_fract_part;

            if (res.skim >= 1.0) {
                skim_int_part = @floor(res.skim);
                res.expected += @as(i64, @intFromFloat(skim_int_part));
                res.skim -= skim_int_part;
            }
        }

        return res;
    }

    pub fn last_improved(self: *Species) i64 {
        return self.age - self.age_of_last_improvement;
    }

    pub fn size(self: *Species) usize {
        return self.organisms.items.len;
    }

    pub fn sort_find_champion(self: *Species) ?*Organism {
        var champ: ?*Organism = null;
        // sort the population (most fit first) and mark for death those after : survival_threshold * pop_size
        std.sort.sort(*Organism, self.organisms.items, {}, fitness_comparison);
        std.mem.reverse(*Organism, self.organisms.items);
        champ = self.organisms.items[0];
        return champ;
    }

    pub fn reproduce(self: *Species, opts: *Options, generation: usize, pop: *Population, sorted_species: []*Species) ![]*Organism {
        // Check for a mistake
        if (self.expected_offspring > 0 and self.organisms.items.len == 0) {
            std.debug.print("attempt to reproduce out of empty species", .{});
            return ReproductionError.CannotReproduceEmptySpecies;
        }

        var prng = std.rand.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.os.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });
        const rand = prng.random();

        // The number of Organisms in the old generation
        var pool_size = self.organisms.items.len;
        // The champion of the 'this' specie is the first element of the specie;
        var the_champ = self.organisms.items[0];

        // the species offspring
        var offspring = std.ArrayList(*Organism).init(self.allocator);

        // Flag the preservation of the champion
        var champ_clone_done = false;

        // Create the designated number of offspring for the Species one at a time
        var count: usize = 0;
        while (count < @as(usize, @intCast(self.expected_offspring))) : (count += 1) {
            var mut_struct_offspring = false;
            var mate_offspring = false;

            var baby: ?*Organism = null;
            if (the_champ.super_champ_offspring > 0) {
                // If we have a super_champ (Population champion), finish off some special clones
                var parent1 = the_champ;
                var new_genome = try parent1.genotype.duplicate(@as(i64, @intCast(count)));

                // Most superchamp offspring will have their connection weights mutated only
                // The last offspring will be an exact duplicate of this super_champ
                // Note: Superchamp offspring only occur with stolen babies!
                //      Settings used for published experiments did not use this
                if (the_champ.super_champ_offspring > 1) {
                    if (rand.float(f64) < 0.8 or opts.mutate_add_link_prob == 0.0) {
                        // Make sure no links get added when the system has link adding disabled
                        _ = try new_genome.mutate_link_weights(opts.weight_mut_power, 1.0, MutatorType.GaussianMutator);
                    } else {
                        // Sometimes we add a link to a superchamp
                        var net = try new_genome.genesis(@as(i64, @intCast(generation)));
                        defer net.deinit();
                        _ = try new_genome.mutate_add_link(pop, opts);
                        mut_struct_offspring = true;
                    }
                }
                // Create the new baby organism
                baby = try Organism.init(self.allocator, 0.0, new_genome, generation);
                if (the_champ.super_champ_offspring == 1 and the_champ.is_population_champion) {
                    baby.?.is_population_champion_child = true;
                    baby.?.highest_fitness = parent1.original_fitness;
                }
                the_champ.super_champ_offspring -= 1;
            } else if (!champ_clone_done and self.expected_offspring > 5) {
                // If we have a Species champion, just clone it
                var parent1 = the_champ;
                var new_genome = try parent1.genotype.duplicate(@as(i64, @intCast(count)));
                champ_clone_done = true;
                // create the new baby organism
                baby = try Organism.init(self.allocator, 0.0, new_genome, generation);
            } else if (rand.float(f64) < opts.mut_only_prob or pool_size == 1) {
                // Apply mutations
                var org_num = rand.uintLessThan(usize, pool_size);
                var parent1 = self.organisms.items[org_num];
                var new_genome = try parent1.genotype.duplicate(@as(i64, @intCast(count)));

                // Do the mutation depending on probabilities of various mutations
                if (rand.float(f64) < opts.mut_add_node_prob) {
                    // Mutate add node
                    _ = try new_genome.mutate_add_node(pop, opts);
                    mut_struct_offspring = true;
                } else if (rand.float(f64) < opts.mut_add_link_prob) {
                    // Mutate add link
                    var net = try new_genome.genesis(@as(i64, @intCast(generation)));
                    defer net.deinit();
                    _ = try new_genome.mutate_add_link(pop, opts);
                    mut_struct_offspring = true;
                } else if (rand.float(f64) < opts.mut_connect_sensors) {
                    mut_struct_offspring = try new_genome.mutate_connect_sensors(pop, opts);
                }

                if (!mut_struct_offspring) {
                    // If we didn't do a structural mutation, we do the other kinds
                    _ = try new_genome.mutate_all_nonstructural(opts);
                }

                // Create the new baby organism
                baby = try Organism.init(self.allocator, 0.0, new_genome, generation);
            } else {
                // Otherwise we should mate
                var org_num = rand.uintLessThan(usize, pool_size);
                var parent1 = self.organisms.items[org_num];

                // choose random parent2
                var parent2: *Organism = undefined;
                if (rand.float(f64) > opts.interspecies_mate_rate) {
                    // Mate within Species
                    org_num = rand.uintLessThan(usize, pool_size);
                    parent2 = self.organisms.items[org_num];
                } else {
                    // Mate outside Species
                    var rand_species = self;

                    // select a random species
                    var give_up: usize = 0;
                    while (rand_species.id == self.id and give_up < 5) : (give_up += 1) {
                        // Choose a random species tending towards better species
                        var rand_mult = rand.float(f64) / 4.0;
                        // This tends to select better species
                        var rand_species_num: usize = @as(usize, @intFromFloat(@floor(rand_mult * @as(f64, @floatFromInt(sorted_species.len)))));
                        rand_species = sorted_species[rand_species_num];
                    }
                    parent2 = rand_species.organisms.items[0];
                }

                // Perform mating based on probabilities of different mating types
                var new_genome: *Genome = undefined;
                if (rand.float(f64) < opts.mate_multipoint_prob) {
                    // mate multipoint baby
                    new_genome = try parent1.genotype.mate_multipoint(parent2.genotype, @as(i64, @intCast(count)), parent1.original_fitness, parent2.original_fitness);
                } else if (rand.float(f64) < opts.mate_multipoint_avg_prob / (opts.mate_multipoint_avg_prob + opts.mate_singlepoint_prob)) {
                    // mate multipoint_avg baby
                    new_genome = try parent1.genotype.mate_multipoint_avg(parent2.genotype, @as(i64, @intCast(count)), parent1.original_fitness, parent2.original_fitness);
                } else {
                    // mate singlepoint baby
                    new_genome = try parent1.genotype.mate_singlepoint(parent2.genotype, @as(i64, @intCast(count)));
                }

                mate_offspring = true;

                // Determine whether to mutate the baby's Genome
                // This is done randomly or if the mom and dad are the same organism
                if (rand.float(f64) > opts.mate_only_prob or parent2.genotype.id == parent1.genotype.id or parent2.genotype.compatability(parent1.genotype, opts) == 0.0) {
                    // Do the mutation depending on probabilities of  various mutations
                    if (rand.float(f64) < opts.mut_add_node_prob) {
                        // mutate_add_node
                        _ = try new_genome.mutate_add_node(pop, opts);
                        mut_struct_offspring = true;
                    } else if (rand.float(f64) < opts.mut_add_link_prob) {
                        // mutate_add_link
                        var net = try new_genome.genesis(@as(i64, @intCast(generation)));
                        defer net.deinit();
                        _ = try new_genome.mutate_add_link(pop, opts);
                        mut_struct_offspring = true;
                    } else if (rand.float(f64) < opts.mut_connect_sensor) {
                        mut_struct_offspring = try new_genome.mutate_connect_sensors(pop, opts);
                    }

                    if (!mut_struct_offspring) {
                        // If we didn't do a structural mutation, we do the other kinds
                        _ = try new_genome.mutate_all_nonstructural(opts);
                    }
                }
                baby = try Organism.init(self.allocator, 0.0, new_genome, generation);
            }
            baby.?.mutation_struct_baby = mut_struct_offspring;
            baby.?.mate_baby = mate_offspring;
            try offspring.append(baby.?);
        }
        return try offspring.toOwnedSlice();
    }
};

pub const ReproductionError = error{
    CannotReproduceEmptySpecies,
} || neat_genome.GenomeError || std.mem.Allocator.Error;

pub fn create_first_species(pop: *Population, baby: *Organism) !void {
    pop.last_species += 1;
    var species = try Species.init_novel(pop.allocator, pop.last_species, true);
    try pop.species.append(species);
    try species.add_organism(baby); // add the new offspring
    baby.species = species; // point offspring to its species
}

pub fn build_test_species_with_organisms(allocator: std.mem.Allocator, id: usize) !*Species {
    var gen = try neat_genome.build_test_genome(allocator, 1);
    defer gen.deinit();
    var sp = try Species.init(allocator, @as(i64, @intCast(id)));
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var gnome_cpy = try gen.duplicate(gen.id);
        var org = try Organism.init(allocator, @as(f64, @floatFromInt(i + 1)) * 5.0 * @as(f64, @floatFromInt(id)), gnome_cpy, id);
        try sp.add_organism(org);
    }
    return sp;
}

test "Species adjust fitness" {
    var allocator = std.testing.allocator;
    var sp = try build_test_species_with_organisms(allocator, 1);
    defer sp.deinit();

    // configuration
    var options = Options{
        .dropoff_age = 5,
        .survival_thresh = 0.5,
        .age_significance = 0.5,
    };
    sp.adjust_fitness(&options);

    // test results
    try std.testing.expect(sp.organisms.items[0].is_champion);
    try std.testing.expect(sp.age_of_last_improvement == 1);
    try std.testing.expect(sp.max_fitness_ever == 15.0);
    try std.testing.expect(sp.organisms.items[2].to_eliminate);
}

test "Species count offspring" {
    var allocator = std.testing.allocator;
    var sp = try build_test_species_with_organisms(allocator, 1);
    defer sp.deinit();
    for (sp.organisms.items, 0..) |o, i| {
        o.expected_offspring = @as(f64, @floatFromInt(i)) * 1.5;
    }
    var res = try sp.count_offspring(0.5);
    defer res.deinit();
    sp.expected_offspring = res.expected;
    try std.testing.expect(sp.expected_offspring == 5);
    try std.testing.expect(res.skim == 0);

    // build and test another species
    var sp2 = try build_test_species_with_organisms(allocator, 2);
    defer sp2.deinit();
    for (sp2.organisms.items, 0..) |o, i| {
        o.expected_offspring = @as(f64, @floatFromInt(i)) * 1.5;
    }
    var res2 = try sp.count_offspring(0.4);
    defer res2.deinit();
    sp2.expected_offspring = res2.expected;
    try std.testing.expect(sp2.expected_offspring == 4);
    try std.testing.expect(res2.skim == 0.9);
}

test "Species compute max fitness" {
    var allocator = std.testing.allocator;
    var sp = try build_test_species_with_organisms(allocator, 1);
    defer sp.deinit();
    var avg_check: f64 = 0;
    for (sp.organisms.items) |o| {
        avg_check += o.fitness;
    }
    avg_check /= @as(f64, @floatFromInt(sp.organisms.items.len));

    var res = try sp.compute_max_and_avg_fitness();
    defer res.deinit();
    try std.testing.expect(res.max == 15.0);
    try std.testing.expect(res.avg == avg_check);
}

test "Species find champion" {
    var allocator = std.testing.allocator;
    var sp = try build_test_species_with_organisms(allocator, 1);
    defer sp.deinit();

    var champ = sp.find_champion();
    try std.testing.expect(champ.?.fitness == 15.0);
}
