const std = @import("std");
const neat_species = @import("species.zig");
const neat_organism = @import("organism.zig");
const neat_genome = @import("genome.zig");
const opt = @import("../opts.zig");
const neat_population = @import("population.zig");

const Options = opt.Options;
const Genome = neat_genome.Genome;
const Population = neat_population.Population;
const Species = neat_species.Species;
const Organism = neat_organism.Organism;
const WaitGroup = std.Thread.WaitGroup;
const logger = @constCast(opt.logger);

pub const ReproductionResult = struct {
    babies_stored: usize = 0,
    babies: ?[]*Organism = null,
    species_id: i64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: i64) !*ReproductionResult {
        var self = try allocator.create(ReproductionResult);
        self.* = .{
            .allocator = allocator,
            .species_id = id,
        };
        return self;
    }

    pub fn deinit(self: *ReproductionResult) void {
        if (self.babies != null) {
            self.allocator.free(self.babies.?);
        }
        self.allocator.destroy(self);
    }
};

pub const SequentialPopulationEpochExecutor = struct {
    // sorted_species sorted to have species with the best fitness score first
    sorted_species: std.ArrayList(*Species),
    best_species_reproduced: bool,
    best_species_id: i64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*SequentialPopulationEpochExecutor {
        var self = try allocator.create(SequentialPopulationEpochExecutor);
        self.* = .{
            .sorted_species = std.ArrayList(*Species).init(allocator),
            .best_species_reproduced = false,
            .best_species_id = undefined,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *SequentialPopulationEpochExecutor) void {
        self.sorted_species.deinit();
        self.allocator.destroy(self);
    }

    pub fn next_epoch(self: *SequentialPopulationEpochExecutor, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, generation: usize, population: *Population) !void {
        try self.prepare_for_reproduction(allocator, rand, opts, generation, population);
        try self.reproduce(allocator, rand, opts, generation, population);
        try self.finalize_reproduction(allocator, opts, population);
    }

    pub fn prepare_for_reproduction(self: *SequentialPopulationEpochExecutor, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, generation: usize, p: *Population) !void {
        // clear executor state from previous run
        self.sorted_species.clearRetainingCapacity();

        // Use Species' ages to modify the objective fitness of organisms in other words, make it more fair for younger
        // species, so they have a chance to take hold and also penalize stagnant species. Then adjust the fitness using
        // the species size to "share" fitness within a species. Then, within each Species, mark for death those below
        // survival_thresh * average
        for (p.species.items) |sp| {
            sp.adjust_fitness(opts);
        }

        // find and remove species unable to produce offspring due to fitness stagnation
        try p.purge_zero_offspring_species(allocator, generation);

        // stick species pointers into new Species list for sorting
        try self.sorted_species.ensureTotalCapacityPrecise(p.species.items.len);
        try self.sorted_species.resize(p.species.items.len);

        @memcpy(self.sorted_species.items, p.species.items);
        // Sort the Species by max original fitness of its first organism
        std.mem.sort(*Species, self.sorted_species.items, {}, fitness_comparison);
        std.mem.reverse(*Species, self.sorted_species.items);

        var curr_species = self.sorted_species.items[0];
        // Used in debugging to see why (if) best species dies
        self.best_species_id = curr_species.id;

        // check for population-level stagnation
        curr_species.organisms.items[0].is_population_champion = true; // DEBUG marker of the best of pop
        if (curr_species.organisms.items[0].og_fitness > p.highest_fitness) {
            p.highest_fitness = curr_species.organisms.items[0].og_fitness;
            p.epoch_highest_last_changed = 0;
        } else {
            p.epoch_highest_last_changed += 1;
        }
        // check for stagnation; if found, perform delta-coding
        if (p.epoch_highest_last_changed >= opts.dropoff_age + 5) {
            // Population stagnated - trying to fix it by delta coding
            p.delta_coding(self.sorted_species.items, opts);
        } else if (opts.babies_stolen > 0) {
            // STOLEN BABIES: The system can take expected offspring away from worse species and give them
            // to superior species depending on the system parameter BabiesStolen (when BabiesStolen > 0)
            try p.give_babies_to_the_best(rand, self.sorted_species.items, opts);
        }
        // Kill off all Organisms marked for death. The remainder will be allowed to reproduce.
        try p.purge_organisms(allocator);
    }

    pub fn reproduce(self: *SequentialPopulationEpochExecutor, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, generation: usize, p: *Population) !void {
        logger.info("POPULATION: Start Sequential Reproduction Cycle >>>>>", .{}, @src());

        // Perform reproduction. Reproduction is done on a per-Species basis
        var offspring = try std.ArrayList(*Organism).initCapacity(allocator, opts.pop_size);
        defer offspring.deinit();

        for (p.species.items) |sp| {
            var rep_offspring = try sp.reproduce(allocator, rand, opts, generation, p, self.sorted_species.items);
            defer allocator.free(rep_offspring);
            if (sp.id == self.best_species_id) {
                // store flag if best species reproduced - it will be used to determine if best species
                // produced offspring before died
                self.best_species_reproduced = true;
            }
            // store offspring
            offspring.appendSliceAssumeCapacity(rep_offspring);
        }

        // sanity check - make sure that population size keep the same
        if (offspring.items.len != opts.pop_size) {
            logger.err("progeny size after reproduction cycle dimished, expected: {d}, but got: {d}", .{ opts.pop_size, offspring.items.len }, @src());
            return error.ReproductionPopSizeMismatch;
        }

        try p.speciate(allocator, opts, offspring.items);

        logger.info("POPULATION: >>>>> Reproduction Complete\n", .{}, @src());
    }

    pub fn finalize_reproduction(self: *SequentialPopulationEpochExecutor, allocator: std.mem.Allocator, _: *Options, pop: *Population) !void {
        // Destroy and remove the old generation from the organisms and species
        try pop.purge_old_generation(allocator);

        // Removes all empty Species and age ones that survive.
        // As this happens, create master organism list for the new generation.
        try pop.purge_or_age_species(allocator);

        // Remove the innovations of the current generation
        for (pop.innovations.items) |innov| {
            innov.deinit();
        }
        pop.innovations.clearAndFree();

        // Check to see if the best species died somehow. We don't want this to happen!!!
        try pop.check_best_species_alive(self.best_species_id, self.best_species_reproduced);
    }
};

pub const ParallelPopulationEpochExecutor = struct {
    sequential: *SequentialPopulationEpochExecutor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*ParallelPopulationEpochExecutor {
        var self = try allocator.create(ParallelPopulationEpochExecutor);
        self.* = .{
            .sequential = undefined,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *ParallelPopulationEpochExecutor) void {
        self.allocator.destroy(self);
    }

    pub fn next_epoch(self: *ParallelPopulationEpochExecutor, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, generation: usize, population: *Population) !void {
        self.sequential = try SequentialPopulationEpochExecutor.init(allocator);
        defer self.sequential.deinit();
        try self.sequential.prepare_for_reproduction(allocator, rand, opts, generation, population);

        // Do parallel reproduction
        try self.reproduce(allocator, rand, opts, generation, population);
        logger.info("POPULATION: >>>>> Epoch {d} complete\n", .{generation}, @src());
        try self.sequential.finalize_reproduction(allocator, opts, population);
    }

    pub fn reproduce(self: *ParallelPopulationEpochExecutor, allocator: std.mem.Allocator, rand: std.rand.Random, opts: *Options, generation: usize, pop: *Population) !void {
        std.debug.print("POPULATION: Start Parallel Reproduction Cycle >>>>>\n", .{});

        // Perform reproduction. Reproduction is done on a per-Species basis
        var sp_num = pop.species.items.len;
        var ctx = try WorkerCtx.init(allocator, sp_num);
        defer ctx.deinit();
        var pool: std.Thread.Pool = undefined;
        try pool.init(.{ .allocator = allocator });
        defer pool.deinit();

        var wait_group: WaitGroup = .{};
        for (pop.species.items, 0..) |species, i| {
            wait_group.start();
            try pool.spawn(parallel_executor_worker_fn, .{ rand, ctx, &wait_group, species, opts, generation, pop, self.sequential.sorted_species.items, i });
        }
        wait_group.wait();

        // read reproduction results, instantiate progeny and speciate over population
        var babies = std.ArrayList(*Organism).init(allocator);
        defer babies.deinit();
        for (ctx.res) |res| {
            if (res.babies != null) {
                if (res.species_id == self.sequential.best_species_id and res.babies.?.len > 0) {
                    // store flag if best species reproduced - it will be used to determine if best species
                    // produced offspring before died
                    self.sequential.best_species_reproduced = true;
                }
                for (res.babies.?) |baby| {
                    try babies.append(baby);
                }
            }
        }

        // sanity check - make sure that population size keep the same
        if (babies.items.len != opts.pop_size) {
            std.debug.print("progeny size after reproduction cycle dimished, expected: [{d}], but got: [{d}]", .{ opts.pop_size, babies.items.len });
            return error.ReproductionPopSizeMismatch;
        }

        // speciate fresh progeny
        try pop.speciate(allocator, opts, babies.items);

        std.debug.print("POPULATION: >>>>> Reproduction Complete\n", .{});
    }
};

pub const WorkerCtx = struct {
    allocator: std.mem.Allocator,
    mu: std.Thread.Mutex = .{},
    res: []*ReproductionResult,

    pub fn init(allocator: std.mem.Allocator, count: usize) !*WorkerCtx {
        var self = try allocator.create(WorkerCtx);
        var res = try allocator.alloc(*ReproductionResult, count);
        for (res) |*r| r.* = try ReproductionResult.init(allocator, 0);
        self.* = .{
            .allocator = allocator,
            .res = res,
        };
        return self;
    }

    pub fn deinit(self: *WorkerCtx) void {
        for (self.res) |res| {
            res.deinit();
        }
        self.allocator.free(self.res);
        self.allocator.destroy(self);
    }
};

fn parallel_executor_worker_fn(rand: std.rand.Random, ctx: *WorkerCtx, wg: *WaitGroup, species: *Species, opts: *Options, generation: usize, pop: *Population, sorted_species: []*Species, idx: usize) void {
    defer wg.finish();
    var res = species.reproduce(ctx.allocator, rand, opts, generation, pop, sorted_species) catch null;
    if (res != null) {
        ctx.mu.lock();
        defer ctx.mu.unlock();
        ctx.res[idx].babies = res;
    }
}

pub fn fitness_comparison(context: void, a: *Species, b: *Species) bool {
    _ = context;
    const org1 = a.organisms.items[0];
    const org2 = b.organisms.items[0];
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

fn sequential_executor_next_epoch(allocator: std.mem.Allocator, rand: std.rand.Random, pop: *Population, opts: *Options) !bool {
    var executor = try SequentialPopulationEpochExecutor.init(allocator);
    defer executor.deinit();
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        executor.next_epoch(allocator, rand, opts, i, pop) catch {
            return false;
        };
    }
    return true;
}

fn parallel_executor_next_epoch(allocator: std.mem.Allocator, rand: std.rand.Random, pop: *Population, opts: *Options) !bool {
    var executor = try ParallelPopulationEpochExecutor.init(allocator);
    defer executor.deinit();
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        executor.next_epoch(allocator, rand, opts, i, pop) catch {
            return false;
        };
    }
    return true;
}

test "SequentialPopulationEpochExecutor next epoch" {
    var allocator = std.testing.allocator;

    var in: i64 = 3;
    var out: i64 = 2;
    var nmax: i64 = 15;
    var n: i64 = 3;
    var link_prob: f64 = 0.8;
    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();
    // configuration
    var opts = Options{
        .compat_threshold = 0.5,
        .dropoff_age = 1,
        .pop_size = 30,
        .babies_stolen = 10,
        .recur_only_prob = 0.2,
    };

    var gen = try Genome.init_rand(allocator, rand, 1, in, out, n, nmax, false, link_prob);
    defer gen.deinit();
    var pop = try Population.init(allocator, rand, gen, &opts);
    defer pop.deinit();

    var res = try sequential_executor_next_epoch(allocator, rand, pop, &opts);
    try std.testing.expect(res);
}

test "ParallelPopulationEpochExecutor next epoch" {
    var threadsafe_test_alloc = std.heap.ThreadSafeAllocator{ .child_allocator = std.testing.allocator };
    const allocator = threadsafe_test_alloc.allocator();
    var in: i64 = 3;
    var out: i64 = 2;
    var nmax: i64 = 15;
    var n: i64 = 3;
    var link_prob: f64 = 0.8;

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    // configuration
    var opts = Options{
        .compat_threshold = 0.5,
        .dropoff_age = 1,
        .pop_size = 30,
        .babies_stolen = 10,
        .recur_only_prob = 0.2,
    };

    var gen = try Genome.init_rand(allocator, rand, 1, in, out, n, nmax, false, link_prob);
    defer gen.deinit();
    var pop = try Population.init(allocator, rand, gen, &opts);
    defer pop.deinit();

    var res = try parallel_executor_next_epoch(allocator, rand, pop, &opts);
    try std.testing.expect(res);
}
