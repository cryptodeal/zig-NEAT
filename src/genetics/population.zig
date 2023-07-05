const std = @import("std");
const opt = @import("../opts.zig");
const neat_species = @import("species.zig");
const neat_organism = @import("organism.zig");
const neat_genome = @import("genome.zig");
const common = @import("common.zig");
const neat_innovation = @import("innovation.zig");

const Species = neat_species.Species;
const create_first_species = neat_species.create_first_species;
const Organism = neat_organism.Organism;
const Innovation = neat_innovation.Innovation;
const Options = opt.Options;
const Genome = neat_genome.Genome;
const MutatorType = common.MutatorType;

pub const Population = struct {
    // species in the population; should comprise all the Genomes
    species: std.ArrayList(*Species),
    // organisms in the population
    organisms: std.ArrayList(*Organism),
    // highest species number
    last_species: i64 = 0,
    // integer that, when greater than 0, indicates when the first winner appeared
    winner_gen: usize = 0,
    // last generation run
    final_gen: usize = undefined,

    // stagnation detector
    highest_fitness: f64 = 0.0,
    epoch_highest_last_changed: usize = 0,

    // fitness characteristics
    mean_fitness: f64 = undefined,
    variance: f64 = undefined,
    standard_deviation: f64 = undefined,

    // holds genetic innovations of most recent population
    innovations: std.ArrayList(*Innovation),
    // next innovation number for the population
    next_innov_number: std.atomic.Atomic(i64) = undefined,
    // next node id for population
    next_node_id: std.atomic.Atomic(i64) = undefined,

    // mutex to guard against concurrent modifications
    mutex: std.Thread.Mutex = .{},

    // threadsafe allocator; enabling concurrent allocations/deallocations
    threadsafe_allocator: std.heap.ThreadSafeAllocator,
    allocator: std.mem.Allocator,
    org_mem_pool: ?std.heap.MemoryPoolExtra(Organism, .{}) = null,

    pub fn raw_init(allocator: std.mem.Allocator) !*Population {
        var multi_thread_alloc = std.heap.ThreadSafeAllocator{ .child_allocator = allocator };
        var self = try allocator.create(Population);
        self.* = .{
            .threadsafe_allocator = multi_thread_alloc,
            .allocator = allocator,
            .species = std.ArrayList(*Species).init(allocator),
            .organisms = std.ArrayList(*Organism).init(allocator),
            .innovations = std.ArrayList(*Innovation).init(allocator),
        };
        return self;
    }

    pub fn init(allocator: std.mem.Allocator, g: *Genome, opts: *Options) !*Population {
        if (opts.pop_size <= 0) {
            std.debug.print("wrong population size in the context: {d}\n", .{opts.pop_size});
            return error.InvalidPopulationSize;
        }
        var self = try Population.raw_init(allocator);
        try self.spawn(g, opts);
        return self;
    }

    pub fn init_random(allocator: std.mem.Allocator, in: usize, out: usize, max_hidden: usize, recurrent: bool, link_prob: f64, opts: *Options) !*Population {
        if (opts.pop_size <= 0) {
            std.debug.print("wrong population size in the options: {d}", .{opts.pop_size});
            return error.InvalidPopulationSize;
        }
        var prng = std.rand.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.os.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });
        const rand = prng.random();
        var self = try Population.raw_init(allocator);
        var count: usize = 0;
        while (count < opts.pop_size) : (count += 1) {
            var gen = try Genome.init_rand(self.allocator, @as(i64, @intCast(count)), @as(i64, @intCast(in)), @as(i64, @intCast(out)), rand.intRangeLessThan(i64, 0, @as(i64, @intCast(max_hidden))), @as(i64, @intCast(max_hidden)), recurrent, link_prob);
            var org = try Organism.init(self.allocator, 0.0, gen, 1);
            try self.organisms.append(org);
        }
        self.next_node_id = std.atomic.Atomic(i64).init(@as(i64, @intCast(in + out + max_hidden + 1)));
        self.next_innov_number = std.atomic.Atomic(i64).init(@as(i64, @intCast((in + out + max_hidden) * (in + out + max_hidden) + 1)));
        try self.speciate(opts, self.organisms.items);
        return self;
    }

    pub fn deinit(self: *Population) void {
        // may need to iterate through each list to free items
        for (self.species.items) |s| {
            s.deinit();
        }
        self.species.deinit();

        self.organisms.deinit();
        for (self.innovations.items) |i| {
            i.deinit();
        }
        self.innovations.deinit();
        self.allocator.destroy(self);
    }

    pub fn verify(self: *Population) !bool {
        for (self.organisms.items) |o| {
            _ = try o.genotype.verify();
        }
        return true;
    }

    pub fn get_next_node_id(self: *Population) i64 {
        var res = self.next_node_id.fetchAdd(1, .Monotonic);
        return res + 1;
    }

    pub fn get_next_innovation_number(self: *Population) i64 {
        var res = self.next_innov_number.fetchAdd(1, .Monotonic);
        return res + 1;
    }

    pub fn store_innovation(self: *Population, innovation: *Innovation) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.innovations.append(innovation);
    }

    pub fn get_innovations(self: *Population) []*Innovation {
        return self.innovations.items;
    }

    pub fn spawn(self: *Population, g: *Genome, opts: *Options) !void {
        var count: usize = 0;
        while (count < opts.pop_size) : (count += 1) {
            // make genome duplicate for new organism
            var new_genome = try g.duplicate(@as(i64, @intCast(count)));
            // introduce initial mutations
            _ = try new_genome.mutate_link_weights(1.0, 1.0, MutatorType.GaussianMutator);
            // create organism for new genome
            var new_organism = try Organism.init(self.allocator, 0, new_genome, 1);
            try self.organisms.append(new_organism);
        }
        // keep track of innovation and node number
        var next_node_id = try g.get_last_node_id();
        self.next_node_id = std.atomic.Atomic(i64).init(next_node_id + 1);
        self.next_innov_number = std.atomic.Atomic(i64).init(try g.get_next_gene_innov_num());
        _ = self.next_innov_number.fetchSub(1, .Monotonic);

        try self.speciate(opts, self.organisms.items);
    }

    pub fn check_best_species_alive(self: *Population, best_species_id: i64, best_species_reproduced: bool) !void {
        var best_ok = false;
        var best_sp_max_fitness: f64 = undefined;
        for (self.species.items) |curr_species| {
            if (curr_species.id == best_species_id) {
                best_ok = true;
                best_sp_max_fitness = curr_species.max_fitness_ever;
            }
        }
        if (!best_ok and !best_species_reproduced) {
            std.debug.print("best species died without offspring\n", .{});
            return error.BestSpeciesDiedWithoutOffspring;
        }
    }

    pub fn speciate(self: *Population, opts: *Options, organisms: []*Organism) !void {
        if (organisms.len == 0) {
            std.debug.print("no organisms to speciate from\n", .{});
            return error.NoOrganismsToSpeciateFrom;
        }

        // step through all given organisms and speciate them within population
        for (organisms) |curr_org| {
            if (self.species.items.len == 0) {
                // create the first species
                try create_first_species(self, curr_org);
            } else {
                if (opts.compat_threshold == 0) {
                    std.debug.print("compatibility threshold is set to ZERO; will not find any compatible species\n", .{});
                    return error.CompatibilityThresholdIsZero;
                }
                // for each organism, search for a species it is compatible with
                var done = false;
                var best_compatible: ?*Species = null;
                var best_compat_value = std.math.inf(f64);
                for (self.species.items) |curr_species| {
                    var comp_org = curr_species.first_organism();
                    // compare current organism with first organism in current species
                    if (comp_org != null) {
                        var curr_compat = curr_org.genotype.compatability(comp_org.?.genotype, opts);
                        if (curr_compat < opts.compat_threshold and curr_compat < best_compat_value) {
                            best_compatible = curr_species;
                            best_compat_value = curr_compat;
                            done = true;
                        }
                    }
                }
                if (best_compatible != null and done) {
                    // found compatible species, so add the current organism to it
                    try best_compatible.?.add_organism(curr_org);
                    // point organism to its species
                    curr_org.species = best_compatible.?;
                } else {
                    // if we didn't find a match, create a new species
                    try create_first_species(self, curr_org);
                }
            }
        }
    }

    pub fn purge_zero_offspring_species(self: *Population, generation: usize) !void {
        _ = generation;
        // used to compute avg fitness over all Organisms
        var total: f64 = 0.0;
        var total_organisms: i64 = @as(i64, @intCast(self.organisms.items.len));

        // go through organisms, adding fitness to compute avg
        for (self.organisms.items) |o| {
            total += o.fitness;
        }

        // avg modified fitness amonst ALL organisms
        var overall_avg = total / @as(f64, @floatFromInt(total_organisms));

        // compute expected number of offspring that can be used only when they accumulate above 1
        if (overall_avg != 0) {
            for (self.organisms.items) |o| {
                o.expected_offspring = o.fitness / overall_avg;
            }
        }

        // The fractional parts of expected offspring that can be used only when they accumulate above 1 for the purposes
        // of counting Offspring
        var skim: f64 = 0.0;
        // precision checking
        var total_expected: i64 = 0;
        for (self.species.items) |sp| {
            var offspring_count = try sp.count_offspring(skim);
            defer offspring_count.deinit();
            sp.expected_offspring = offspring_count.expected;
            skim = offspring_count.skim;
            total_expected += sp.expected_offspring;
        }

        // Need to make up for lost floating point precision in offspring assignment.
        // If we lost precision, give an extra baby to the best Species
        if (total_expected < total_organisms) {
            // find the species expecting the most
            var best_species: ?*Species = null;
            var max_expected: i64 = 0;
            var final_expected: i64 = 0;
            for (self.species.items) |sp| {
                if (sp.expected_offspring >= max_expected) {
                    max_expected = sp.expected_offspring;
                    best_species = sp;
                }
                final_expected += sp.expected_offspring;
            }
            // give the extra offspring to the best species
            if (best_species != null) {
                best_species.?.expected_offspring += 1;
            }
            final_expected += 1;

            // If we still aren't at total, there is a problem. Note that this can happen if a stagnant Species
            // dominates the population and then gets killed off by its age. Then the whole population plummets in
            // fitness. If the average fitness is allowed to hit 0, then we no longer have an average we can use to
            // assign offspring.
            if (final_expected < total_organisms) {
                for (self.species.items) |sp| {
                    sp.expected_offspring = 0;
                }
                if (best_species != null) {
                    best_species.?.expected_offspring = @as(i64, @intCast(total_organisms));
                }
            }
        }
        // Remove stagnated species which can not produce any offspring
        var species_to_keep = std.ArrayList(*Species).init(self.allocator);
        for (self.species.items) |sp| {
            if (sp.expected_offspring > 0) {
                try species_to_keep.append(sp);
            } else {
                // TODO: ensure this frees memory of extinct species
                var new_orgs = std.ArrayList(*Organism).init(self.allocator);
                for (self.organisms.items) |o| {
                    if (o.species.id != sp.id) {
                        try new_orgs.append(o);
                    }
                }
                sp.deinit();
                self.organisms.deinit();
                self.organisms = new_orgs;
            }
        }
        self.species.deinit();
        self.species = species_to_keep;
    }

    // When population stagnation detected the delta coding will be performed in attempt to fix this
    pub fn delta_coding(self: *Population, sorted_species: []*Species, opts: *Options) void {
        self.epoch_highest_last_changed = 0;
        var half_pop = opts.pop_size / 2;

        var curr_species = sorted_species[0];
        if (sorted_species.len > 1) {
            // Assign population to first two species
            curr_species.organisms.items[0].super_champ_offspring = half_pop;
            curr_species.expected_offspring = @as(i64, @intCast(half_pop));
            curr_species.age_of_last_improvement = curr_species.age;

            // process the 2nd species
            curr_species = sorted_species[1];
            // NOTE: PopSize can be odd. That's why we use subtraction below
            curr_species.organisms.items[0].super_champ_offspring = opts.pop_size - half_pop;
            curr_species.expected_offspring = @as(i64, @intCast(opts.pop_size - half_pop));
            curr_species.age_of_last_improvement = curr_species.age;
            // Get rid of all species after the first two
            var i: usize = 2;
            while (i < sorted_species.len) : (i += 1) {
                sorted_species[i].expected_offspring = 0;
            }
        } else {
            curr_species.organisms.items[0].super_champ_offspring = opts.pop_size;
            curr_species.expected_offspring = @as(i64, @intCast(opts.pop_size));
            curr_species.age_of_last_improvement = curr_species.age;
        }
    }

    // The system can take expected offspring away from worse species and give them
    // to superior species depending on the system parameter BabiesStolen (when BabiesStolen > 0)
    pub fn give_babies_to_the_best(_: *Population, sorted_species: []*Species, opts: *Options) !void {
        var prng = std.rand.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.os.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });
        const rand = prng.random();

        // Babies taken from the bad species and given to the champs
        var stolen_babies: usize = 0;

        // Take away a constant number of expected offspring from the worst few species
        var i: i64 = @as(i64, @intCast(sorted_species.len - 1));
        while (i >= 0 and stolen_babies < opts.babies_stolen) : (i -= 1) {
            var curr_species = sorted_species[@as(usize, @intCast(i))];
            if (curr_species.age > 5 and curr_species.expected_offspring > 2) {
                if (curr_species.expected_offspring - 1 >= opts.babies_stolen - stolen_babies) {
                    // This species has enough to finish off the stolen pool
                    curr_species.expected_offspring -= @as(i64, @intCast(opts.babies_stolen - stolen_babies));
                    stolen_babies = opts.babies_stolen;
                } else {
                    // Not enough here to complete the pool of stolen
                    stolen_babies += @as(usize, @intCast(curr_species.expected_offspring - 1));
                    curr_species.expected_offspring = 1;
                }
            }
        }

        // Mark the best champions of the top species to be the super champs who will take on the extra
        // offspring for cloning or mutant cloning.
        // Determine the exact number that will be given to the top three.
        // They will get, in order, 1/5 1/5 and 1/10 of the stolen babies
        var stolen_blocks = [3]usize{ opts.babies_stolen / 5, opts.babies_stolen / 5, opts.babies_stolen / 10 };
        var block_idx: usize = 0;
        for (sorted_species) |curr_species| {
            if (curr_species.last_improved() > opts.dropoff_age) {
                // Don't give a chance to dying species even if they are champs
                continue;
            }

            if (block_idx < 3 and stolen_babies >= stolen_blocks[block_idx]) {
                // Give stolen babies to the top three in 1/5 1/5 and 1/10 ratios
                curr_species.organisms.items[0].super_champ_offspring = stolen_blocks[block_idx];
                curr_species.expected_offspring += @as(i64, @intCast(stolen_blocks[block_idx]));
                stolen_babies -= stolen_blocks[block_idx];
            } else if (block_idx >= 3) {
                // Give stolen to the rest in random ratios
                if (rand.float(f64) > 0.1) {
                    // Randomize a little which species get boosted by a super champ
                    if (stolen_babies > 3) {
                        curr_species.organisms.items[0].super_champ_offspring = 3;
                        curr_species.expected_offspring += 3;
                        stolen_babies -= 3;
                    } else {
                        curr_species.organisms.items[0].super_champ_offspring = stolen_babies;
                        curr_species.expected_offspring += @as(i64, @intCast(stolen_babies));
                        stolen_babies = 0;
                    }
                }

                if (stolen_babies <= 0) {
                    break;
                }
                block_idx += 1;
            }
        }
        // If any stolen babies aren't taken, give them to species #1's champ
        if (stolen_babies > 0) {
            var curr_species = sorted_species[0];
            curr_species.organisms.items[0].super_champ_offspring += stolen_babies;
            curr_species.expected_offspring += @as(i64, @intCast(stolen_babies));
        }
    }

    // Purge from population all organisms marked to be eliminated
    pub fn purge_organisms(self: *Population) !void {
        var org_to_keep = std.ArrayList(*Organism).init(self.allocator);
        for (self.organisms.items) |org| {
            if (org.to_eliminate) {
                // Remove the organism from its Species
                try org.species.remove_organism(org);
            } else {
                try org_to_keep.append(org);
            }
        }
        self.organisms.deinit();
        self.organisms = org_to_keep;
    }

    // Destroy and remove the old generation of the organisms and of the species
    pub fn purge_old_generation(self: *Population) !void {
        for (self.organisms.items) |org| {
            // Remove the organism from its Species
            try org.species.remove_organism(org);
        }

        self.organisms.deinit();
        self.organisms = std.ArrayList(*Organism).init(self.allocator);
    }

    pub fn purge_or_age_species(self: *Population) !void {
        var org_count: usize = 0;
        var species_to_keep = std.ArrayList(*Species).init(self.allocator);
        for (self.species.items) |curr_species| {
            if (curr_species.organisms.items.len > 0) {
                // Age surviving Species
                if (curr_species.is_novel) {
                    curr_species.is_novel = false;
                } else {
                    curr_species.age += 1;
                }
                // Rebuild master Organism list of population: NUMBER THEM as they are added to the list
                for (curr_species.organisms.items) |org| {
                    org.genotype.id = @as(i64, @intCast(org_count));
                    try self.organisms.append(org);
                    org_count += 1;
                }
                // keep this species
                try species_to_keep.append(curr_species);
            } else {
                // free the species
                curr_species.deinit();
            }
        }
        // Keep only survived species
        self.species.deinit();
        self.species = species_to_keep;
    }
};

test "Population init random" {
    var allocator = std.testing.allocator;
    var in: usize = 3;
    var out: usize = 2;
    var nmax: usize = 5;
    var link_prob: f64 = 0.5;

    // configuration
    var options = Options{
        .compat_threshold = 0.5,
        .pop_size = 10,
    };
    var pop = try Population.init_random(allocator, in, out, nmax, false, link_prob, &options);
    defer pop.deinit();

    try std.testing.expect(pop.organisms.items.len == options.pop_size);
    try std.testing.expect(pop.next_node_id.loadUnchecked() == 11);
    try std.testing.expect(pop.next_innov_number.loadUnchecked() == 101);
    try std.testing.expect(pop.species.items.len > 0);

    for (pop.organisms.items) |org| {
        try std.testing.expect(org.genotype.genes.len > 0);
        try std.testing.expect(org.genotype.nodes.len > 0);
        try std.testing.expect(org.genotype.traits.len > 0);
        try std.testing.expect(org.phenotype != null);
    }
}

test "Population init" {
    var allocator = std.testing.allocator;

    var in: i64 = 3;
    var out: i64 = 2;
    var nmax: i64 = 5;
    var n: i64 = 3;
    var link_prob: f64 = 0.5;

    // configuration
    var options = Options{
        .compat_threshold = 0.5,
        .pop_size = 10,
    };

    var gen = try Genome.init_rand(allocator, 1, in, out, n, nmax, false, link_prob);
    defer gen.deinit();
    var pop = try Population.init(allocator, gen, &options);
    defer pop.deinit();

    try std.testing.expect(pop.organisms.items.len == options.pop_size);
    var last_node_id = try gen.get_last_node_id();
    try std.testing.expect(last_node_id + 1 == pop.next_node_id.loadUnchecked());

    var next_gene_innov_num = try gen.get_next_gene_innov_num();
    try std.testing.expect(next_gene_innov_num - 1 == pop.next_innov_number.loadUnchecked());
    try std.testing.expect(pop.species.items.len == 1);
}
