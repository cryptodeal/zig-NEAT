const std = @import("std");
const orgn = @import("organism.zig");
const neat_population = @import("population.zig");
const neat_common = @import("common.zig");
const neat_genome = @import("genome.zig");
const opt = @import("../opts.zig");

const Organism = orgn.Organism;
const Genome = neat_genome.Genome;
const Options = opt.Options;
const Population = neat_population.Population;
const MutatorType = neat_common.MutatorType;
const fitnessComparison = orgn.fitnessComparison;
const logger = @constCast(opt.logger);

/// Data structure holding the maximal and average
/// fitness of a Species.
pub const MaxAvgFitness = struct {
    max: f64 = 0.0,
    avg: f64 = 0.0,
};

/// Data structure holding the expected offspring count
/// and skim value, which are used when calculating the
/// number of expected offspring per each Species.
pub const OffspringCount = struct {
    expected: i64 = 0,
    skim: f64 = 0.0,
};

/// A Species is a group of similar Organisms.
/// Reproduction primarily takes plaece within the same species,
/// which encourages mating between compatible organisms.
pub const Species = struct {
    /// The Species Id.
    id: i64,
    /// The age of the Species.
    age: i64 = 1,
    /// The maximal fitness of the Species (all-time).
    max_fitness_ever: f64 = undefined,
    /// The count of expected offspring for the Species.
    expected_offspring: i64 = undefined,

    /// Indicates whether the Species is novel.
    is_novel: bool = false,

    /// List of all Organisms in the species; the algorithm keeps it sorted
    /// to have most fit first at beginning of each reproduction cycle.
    organisms: std.ArrayList(*Organism),
    /// Denotes the age of the Species when it's best fitness last improved.
    /// If this is too long ago, the Species will go extinct.
    age_of_last_improvement: i64 = undefined,

    /// Flag used for search optimization.
    is_checked: bool = false,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new Species.
    pub fn init(allocator: std.mem.Allocator, id: i64) !*Species {
        var self = try allocator.create(Species);
        self.* = .{
            .allocator = allocator,
            .id = id,
            .organisms = std.ArrayList(*Organism).init(allocator),
        };
        return self;
    }

    /// Initializes a new Species and flags as being novel.
    pub fn initNovel(allocator: std.mem.Allocator, id: i64, novel: bool) !*Species {
        var self = try Species.init(allocator, id);
        self.is_novel = novel;
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *Species) void {
        for (self.organisms.items) |o| o.deinit();
        self.organisms.deinit();
        self.allocator.destroy(self);
    }

    /// Adds a new Organism to the list of Organisms that comprise this Species.
    pub fn addOrganism(self: *Species, o: *Organism) !void {
        try self.organisms.append(o);
    }

    /// Removes an Organism from the list of Organisms that comprise this Species.
    pub fn removeOrganism(self: *Species, allocator: std.mem.Allocator, org: *Organism) !void {
        var old_orgs = self.organisms;
        var orgs = std.ArrayList(*Organism).init(allocator);
        errdefer orgs.deinit();

        for (old_orgs.items) |o| {
            if (!std.meta.eql(o, org)) {
                try orgs.append(o);
            } else {
                o.deinit();
            }
        }

        if (orgs.items.len != old_orgs.items.len - 1) {
            logger.info("attempt to remove nonexistent Organism from Species with #of organisms: {d}\n", .{old_orgs.items.len}, @src());
            return error.FailedToRemoveOrganism;
        } else {
            old_orgs.deinit();
            self.organisms = orgs;
        }
    }

    /// Used to change the fitness of Organisms in the Species to be higher
    /// for very new Species (to protect them). Divides the fitness by the
    /// size of the Species, so that fitness is "shared" by the Species.
    /// NOTE: Invocation of this method will result in the Species Organisms
    /// sorted by fitness in descending order, i.e. most fit will be first.
    pub fn adjustFitness(self: *Species, opts: *Options) void {
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
        std.mem.sort(*Organism, self.organisms.items, {}, fitnessComparison);
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

    /// Computes the maximal and average fitness of this Species.
    pub fn computeMaxAndAvgFitness(self: *Species) MaxAvgFitness {
        var res = MaxAvgFitness{};
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

    /// Returns most fit organism for this Species.
    pub fn findChampion(self: *Species) ?*Organism {
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

    /// Returns the first Organism in this Species or null if empty.
    pub fn firstOrganism(self: *Species) ?*Organism {
        if (self.organisms.items.len > 0) {
            return self.organisms.items[0];
        } else {
            return null;
        }
    }

    /// Computes the collective offspring the entire Species (the sum of all
    /// Organism's offspring) is assigned. The skim is fractional offspring
    /// left over from a previous Species that was counted. These fractional
    /// parts are kept until they add up to 1. Returns the whole offspring
    /// count for this Species as well as fractional offspring left after
    /// computation (skim).
    pub fn countOffspring(self: *Species, skim: f64) !OffspringCount {
        var org_off_int_part: i64 = undefined;
        var org_off_fract_part: f64 = undefined;
        var skim_int_part: f64 = undefined;

        var res = OffspringCount{};
        res.expected = 0;
        res.skim = skim;

        for (self.organisms.items) |o| {
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

    /// Compute the number of Generations since the Species'
    /// fitness last improved.
    pub fn lastImproved(self: *Species) i64 {
        return self.age - self.age_of_last_improvement;
    }

    /// Returns the size of this Species;
    /// i.e. the number of Organisms belonging to it.
    pub fn size(self: *Species) usize {
        return self.organisms.items.len;
    }

    /// Sorts the Organisms in this Species by fitness and returns
    /// the Organism with the highest fitness score.
    pub fn sortFindChampion(self: *Species) ?*Organism {
        var champ: ?*Organism = null;
        // sort the population (most fit first) and mark for death those after : survival_threshold * pop_size
        std.mem.sort(*Organism, self.organisms.items, {}, fitnessComparison);
        std.mem.reverse(*Organism, self.organisms.items);
        champ = self.organisms.items[0];
        return champ;
    }

    /// Performs mating and mutation to form the next Generation.
    /// The sorted_species is ordered to have best Species in the beginning.
    /// Returns list of baby Organisms as a result of reproduction of all
    /// organisms in this species.
    pub fn reproduce(self: *Species, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, generation: usize, pop: *Population, sorted_species: []*Species) ![]*Organism {
        // Check for a mistake
        if (self.expected_offspring > 0 and self.organisms.items.len == 0) {
            logger.err("attempt to reproduce out of empty species", .{}, @src());
            return ReproductionError.CannotReproduceEmptySpecies;
        }

        // The number of Organisms in the old generation
        var pool_size = self.organisms.items.len;
        // The champion of the 'this' specie is the first element of the specie;
        var the_champ = self.organisms.items[0];

        // the species offspring
        var offspring = std.ArrayList(*Organism).init(allocator);

        // Flag the preservation of the champion
        var champ_clone_done = false;

        // Create the designated number of offspring for the Species one at a time
        var count: usize = 0;
        logger.debug("SPECIES ID: {d} ---- Expected Offspring #{d}", .{ self.id, self.expected_offspring }, @src());
        while (count < @as(usize, @intCast(self.expected_offspring))) : (count += 1) {
            logger.debug("SPECIES: Offspring #{d} from {d}, (species: {d})", .{ count, self.expected_offspring, self.id }, @src());
            var mut_struct_offspring = false;
            var mate_offspring = false;

            // debug trap
            if (self.expected_offspring > opts.pop_size) {
                logger.warn("SPECIES: Species [{d}] expected offspring: {d} exceeds population size limit: {d}", .{ self.id, self.expected_offspring, opts.pop_size }, @src());
            }

            var baby: ?*Organism = null;
            if (the_champ.super_champ_offspring > 0) {
                logger.debug("SPECIES: Reproduce super champion", .{}, @src());

                // If we have a super_champ (Population champion), finish off some special clones
                var mom = the_champ;
                var new_genome = try mom.genotype.duplicate(allocator, @as(i64, @intCast(count)));

                // Most superchamp offspring will have their connection weights mutated only
                // The last offspring will be an exact duplicate of this super_champ
                // Note: Superchamp offspring only occur with stolen babies!
                //      Settings used for published experiments did not use this
                if (the_champ.super_champ_offspring > 1) {
                    if (rand.float(f64) < 0.8 or opts.mut_add_link_prob == 0.0) {
                        // Make sure no links get added when the system has link adding disabled
                        _ = try new_genome.mutateLinkWeights(rand, opts.weight_mut_power, 1.0, MutatorType.GaussianMutator);
                    } else {
                        // Sometimes we add a link to a superchamp
                        _ = try new_genome.genesis(allocator, @as(i64, @intCast(generation)));
                        _ = try new_genome.mutateAddLink(allocator, rand, pop, opts);
                        mut_struct_offspring = true;
                    }
                }

                // Create the new baby organism
                baby = try Organism.init(allocator, 0.0, new_genome, generation);
                if (the_champ.super_champ_offspring == 1 and the_champ.is_population_champion) {
                    baby.?.is_population_champion_child = true;
                    baby.?.highest_fitness = mom.og_fitness;
                }
                the_champ.super_champ_offspring -= 1;
            } else if (!champ_clone_done and self.expected_offspring > 5) {
                logger.debug("SPECIES: Clone species champion", .{}, @src());
                // If we have a Species champion, just clone it
                var mom = the_champ; // Mom is the champ
                var new_genome = try mom.genotype.duplicate(allocator, @as(i64, @intCast(count)));
                // Baby is just like mom
                champ_clone_done = true;
                // create the new baby organism
                baby = try Organism.init(allocator, 0.0, new_genome, generation);
            } else if (rand.float(f64) < opts.mut_only_prob or pool_size == 1) {
                logger.debug("SPECIES: Reproduce by applying random mutation:", .{}, @src());
                // Apply mutations
                var org_num = rand.uintLessThan(usize, pool_size); // select random mom
                var mom = self.organisms.items[org_num];
                var new_genome = try mom.genotype.duplicate(allocator, @as(i64, @intCast(count)));

                // Do the mutation depending on probabilities of various mutations
                if (rand.float(f64) < opts.mut_add_node_prob) {
                    logger.debug("SPECIES: ---> mutateAddNode", .{}, @src());

                    // Mutate add node
                    _ = try new_genome.mutateAddNode(allocator, rand, pop, opts);
                    mut_struct_offspring = true;
                } else if (rand.float(f64) < opts.mut_add_link_prob) {
                    logger.debug("SPECIES: ---> mutateAddLink", .{}, @src());

                    // Mutate add link
                    _ = try new_genome.genesis(allocator, @as(i64, @intCast(generation)));
                    _ = try new_genome.mutateAddLink(allocator, rand, pop, opts);
                    mut_struct_offspring = true;
                } else if (rand.float(f64) < opts.mut_connect_sensors) {
                    logger.debug("SPECIES: ---> mutateConnectSensors", .{}, @src());

                    mut_struct_offspring = try new_genome.mutateConnectSensors(allocator, rand, pop, opts);
                }

                if (!mut_struct_offspring) {
                    logger.debug("SPECIES: ---> mutateAllNonstructural", .{}, @src());

                    // If we didn't do a structural mutation, we do the other kinds
                    _ = try new_genome.mutateAllNonstructural(rand, opts);
                }

                // Create the new baby organism
                baby = try Organism.init(allocator, 0.0, new_genome, generation);
            } else {
                logger.debug("SPECIES: Reproduce by mating:", .{}, @src());

                // Otherwise we should mate
                var org_num = rand.uintLessThan(usize, pool_size); // select random mom
                var mom = self.organisms.items[org_num];

                // choose random dad
                var dad: *Organism = undefined;
                if (rand.float(f64) > opts.interspecies_mate_rate) {
                    logger.debug("SPECIES: ---> mate within species", .{}, @src());

                    // Mate within Species
                    org_num = rand.uintLessThan(usize, pool_size);
                    dad = self.organisms.items[org_num];
                } else {
                    logger.debug("SPECIES: ---> mate outside species", .{}, @src());

                    // Mate outside Species
                    var rand_species = self;

                    // select a random species
                    var give_up: usize = 0;
                    while (rand_species.id == self.id and give_up < 5) : (give_up += 1) {
                        // Choose a random species tending towards better species
                        var rand_mult = rand.float(f64) / 4;
                        // This tends to select better species
                        var rand_species_num: usize = @as(usize, @intFromFloat(@floor(rand_mult * @as(f64, @floatFromInt(sorted_species.len)))));
                        rand_species = sorted_species[rand_species_num];
                    }
                    dad = rand_species.organisms.items[0];
                }

                // Perform mating based on probabilities of different mating types
                var new_genome: *Genome = undefined;
                if (rand.float(f64) < opts.mate_multipoint_prob) {
                    logger.debug("SPECIES: ------> mateMultipoint", .{}, @src());

                    // mate multipoint baby
                    new_genome = try mom.genotype.mateMultipoint(allocator, rand, dad.genotype, @as(i64, @intCast(count)), mom.og_fitness, dad.og_fitness);
                } else if (rand.float(f64) < opts.mate_multipoint_avg_prob / (opts.mate_multipoint_avg_prob + opts.mate_singlepoint_prob)) {
                    logger.debug("SPECIES: ------> mateMultipointAvg", .{}, @src());

                    // mate multipoint_avg baby
                    new_genome = try mom.genotype.mateMultipointAvg(allocator, rand, dad.genotype, @as(i64, @intCast(count)), mom.og_fitness, dad.og_fitness);
                } else {
                    logger.debug("SPECIES: ------> mateSinglepoint", .{}, @src());

                    // mate singlepoint baby
                    new_genome = try mom.genotype.mateSinglepoint(allocator, rand, dad.genotype, @as(i64, @intCast(count)));
                }

                mate_offspring = true;

                // Determine whether to mutate the baby's Genome
                // This is done randomly or if the mom and dad are the same organism
                if (rand.float(f64) > opts.mate_only_prob or dad.genotype.id == mom.genotype.id or dad.genotype.compatability(mom.genotype, opts) == 0.0) {
                    logger.debug("SPECIES: ------> Mutate baby genome:", .{}, @src());

                    // Do the mutation depending on probabilities of  various mutations
                    if (rand.float(f64) < opts.mut_add_node_prob) {
                        logger.debug("SPECIES: ---------> mutateAddNode", .{}, @src());

                        // mutateAddNode
                        _ = try new_genome.mutateAddNode(allocator, rand, pop, opts);
                        mut_struct_offspring = true;
                    } else if (rand.float(f64) < opts.mut_add_link_prob) {
                        logger.debug("SPECIES: ---------> mutateAddLink", .{}, @src());

                        // mutateAddLink
                        _ = try new_genome.genesis(allocator, @as(i64, @intCast(generation)));
                        _ = try new_genome.mutateAddLink(allocator, rand, pop, opts);
                        mut_struct_offspring = true;
                    } else if (rand.float(f64) < opts.mut_connect_sensors) {
                        logger.debug("SPECIES: ---------> mutateConnectSensors", .{}, @src());
                        mut_struct_offspring = try new_genome.mutateConnectSensors(allocator, rand, pop, opts);
                    }

                    if (!mut_struct_offspring) {
                        logger.debug("SPECIES: ---------> mutateAllNonstructural", .{}, @src());
                        // If we didn't do a structural mutation, we do the other kinds
                        _ = try new_genome.mutateAllNonstructural(rand, opts);
                    }
                }
                baby = try Organism.init(allocator, 0.0, new_genome, generation);
            } // end else
            baby.?.mutation_struct_baby = mut_struct_offspring;
            baby.?.mate_baby = mate_offspring;
            try offspring.append(baby.?);
        } // end for (count == 0)
        return offspring.toOwnedSlice();
    }
};

fn ogFitnessComparison(context: void, a: *Species, b: *Species) bool {
    _ = context;
    const org1: Organism = a.organisms.items[0].*;
    const org2: Organism = b.organisms.items[0].*;
    if (org1.og_fitness < org2.og_fitness) {
        // try to promote most fit species
        return true; // Lower fitness is less
    } else if (org1.og_fitness == org2.og_fitness) {
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

pub const ReproductionError = error{
    CannotReproduceEmptySpecies,
} || neat_genome.GenomeError || std.mem.Allocator.Error;

pub fn createFirstSpecies(allocator: std.mem.Allocator, pop: *Population, baby: *Organism) !void {
    pop.last_species += 1;
    var species = try Species.initNovel(allocator, pop.last_species, true);
    try pop.species.append(species);
    try species.addOrganism(baby); // add the new offspring
    baby.species = species; // point offspring to its species
}

pub fn buildTestSpeciesWithOrganisms(allocator: std.mem.Allocator, id: usize) !*Species {
    var gen = try neat_genome.buildTestGenome(allocator, 1);
    defer gen.deinit();
    var sp = try Species.init(allocator, @as(i64, @intCast(id)));
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var gnome_cpy = try gen.duplicate(allocator, gen.id);
        var org = try Organism.init(allocator, @as(f64, @floatFromInt(i + 1)) * 5.0 * @as(f64, @floatFromInt(id)), gnome_cpy, id);
        try sp.addOrganism(org);
    }
    return sp;
}

test "Species adjust fitness" {
    var allocator = std.testing.allocator;
    var sp = try buildTestSpeciesWithOrganisms(allocator, 1);
    defer sp.deinit();

    // configuration
    var options = Options{
        .dropoff_age = 5,
        .survival_thresh = 0.5,
        .age_significance = 0.5,
    };
    sp.adjustFitness(&options);

    // test results
    try std.testing.expect(sp.organisms.items[0].is_champion);
    try std.testing.expect(sp.age_of_last_improvement == 1);
    try std.testing.expect(sp.max_fitness_ever == 15.0);
    try std.testing.expect(sp.organisms.items[2].to_eliminate);
}

test "Species count offspring" {
    var allocator = std.testing.allocator;
    var sp = try buildTestSpeciesWithOrganisms(allocator, 1);
    defer sp.deinit();
    for (sp.organisms.items, 0..) |o, i| {
        o.expected_offspring = @as(f64, @floatFromInt(i)) * 1.5;
    }
    var res = try sp.countOffspring(0.5);
    sp.expected_offspring = res.expected;
    try std.testing.expect(sp.expected_offspring == 5);
    try std.testing.expect(res.skim == 0);

    // build and test another species
    var sp2 = try buildTestSpeciesWithOrganisms(allocator, 2);
    defer sp2.deinit();
    for (sp2.organisms.items, 0..) |o, i| {
        o.expected_offspring = @as(f64, @floatFromInt(i)) * 1.5;
    }
    res = try sp.countOffspring(0.4);
    sp2.expected_offspring = res.expected;
    try std.testing.expect(sp2.expected_offspring == 4);
    try std.testing.expect(res.skim == 0.9);
}

test "Species compute max fitness" {
    var allocator = std.testing.allocator;
    var sp = try buildTestSpeciesWithOrganisms(allocator, 1);
    defer sp.deinit();
    var avg_check: f64 = 0;
    for (sp.organisms.items) |o| {
        avg_check += o.fitness;
    }
    avg_check /= @as(f64, @floatFromInt(sp.organisms.items.len));

    var res = sp.computeMaxAndAvgFitness();
    try std.testing.expect(res.max == 15.0);
    try std.testing.expect(res.avg == avg_check);
}

test "Species find champion" {
    var allocator = std.testing.allocator;
    var sp = try buildTestSpeciesWithOrganisms(allocator, 1);
    defer sp.deinit();

    var champ = sp.findChampion();
    try std.testing.expect(champ.?.fitness == 15.0);
}

test "Species remove organism" {
    var allocator = std.testing.allocator;
    var sp = try buildTestSpeciesWithOrganisms(allocator, 1);
    defer sp.deinit();

    // test remove
    var size = sp.organisms.items.len;
    try sp.removeOrganism(allocator, sp.organisms.items[0]);
    try std.testing.expect(sp.organisms.items.len == size - 1);

    // test fail to remove
    size = sp.organisms.items.len;
    var gen = try neat_genome.buildTestGenome(allocator, 1);
    var org = try Organism.init(allocator, 6.0, gen, 1);
    defer org.deinit();
    try std.testing.expectError(error.FailedToRemoveOrganism, sp.removeOrganism(allocator, org));
    try std.testing.expect(sp.organisms.items.len == size);
}

test "Species reproduce" {
    var allocator = std.testing.allocator;
    var in: i64 = 3;
    var out: i64 = 2;
    var nmax: i64 = 15;
    var n: i64 = 3;
    var link_prob: f64 = 0.8;

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    // configuration
    var options = Options{
        .dropoff_age = 5,
        .survival_thresh = 0.5,
        .age_significance = 0.5,
        .pop_size = 30,
        .compat_threshold = 0.6,
    };
    var gen = try Genome.initRand(allocator, rand, 1, in, out, n, nmax, false, link_prob);
    defer gen.deinit();
    var pop = try Population.init(allocator, rand, gen, &options);
    defer pop.deinit();

    // stick the Species pointers into a new Species list for sorting
    var sorted_species = try allocator.alloc(*Species, pop.species.items.len);
    defer allocator.free(sorted_species);
    @memcpy(sorted_species, pop.species.items);

    // sort the Species by max original fitness of its first organism
    std.mem.sort(*Species, sorted_species, {}, ogFitnessComparison);

    pop.species.items[0].expected_offspring = 11;

    var babies = try pop.species.items[0].reproduce(allocator, rand, &options, 1, pop, sorted_species);
    defer allocator.free(babies);
    defer for (babies) |b| {
        b.deinit();
    };
    try std.testing.expect(babies.len == pop.species.items[0].expected_offspring);
}
