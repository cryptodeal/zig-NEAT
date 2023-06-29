const std = @import("std");
const neat_species = @import("species.zig");
const neat_organism = @import("organism.zig");
const opt = @import("../opt.zig");
const neat_population = @import("population.zig");

const Options = opt.Options;
const Population = neat_population.Population;
const Species = neat_species.Species;
const Organism = neat_organism.Organism;
const WaitGroup = std.Thread.WaitGroup;

pub const ReproductionResult = struct {
    babies_stored: usize,
    babies: ?[]*Organism,
    species_id: i64,

    pub fn init(allocator: std.mem.Allocator, id: i64) *ReproductionResult {
        var self = try allocator.create(ReproductionResult);
        self.* = .{
            .babies_stored = 0,
            .babies = undefined,
            .species_id = id,
        };
        return self;
    }

    pub fn deinit(self: *ReproductionResult) void {
        self.allocator.destroy(self);
    }
};

pub const SequentialPopulationEpochExecutor = struct {
    // sorted_species sorted to have species with the best fitness score first
    sorted_species: std.ArrayList(*Species),
    best_species_reproduced: bool,
    best_species_id: i64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) *SequentialPopulationEpochExecutor {
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

    pub fn next_epoch(self: *SequentialPopulationEpochExecutor, opts: *Options, generation: usize, population: *Population) !void {
        try self.prepare_for_reproduction(opts, generation, population);
    }

    pub fn prepare_for_reproduction(self: *SequentialPopulationEpochExecutor, opts: *Options, generation: usize, p: *Population) !void {
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
        p.purge_zero_offspring_species(generation);

        // stick species pointers into new Species list for sorting
        try self.sorted_species.ensureTotalCapacity(p.species.items.len);
        @memcpy(self.sorted_species.items, p.species.items);

        // Sort the Species by max original fitness of its first organism
        std.mem.sort(*Species, self.sorted_species.items, {}, fitness_comparison);
        std.mem.reverse(*Species, self.organisms.items);

        // Used in debugging to see why (if) best species dies
        self.best_species_id = self.sorted_species.items[0].id;

        var curr_species = self.sorted_species.items[0];
        // Used in debugging to see why (if) best species dies
        self.best_species_id = curr_species.id;

        // check for population-level stagnation
        curr_species.organisms.items[0].is_population_champion = true; // DEBUG marker of the best of pop
        if (curr_species.organisms.items[0].original_fitness > p.highest_fitness) {
            p.highest_fitness = curr_species.organisms.items[0].original_fitness;
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
            p.give_babies_to_the_best(self.sorted_species.items, opts);
        }
        // Kill off all Organisms marked for death. The remainder will be allowed to reproduce.
        try p.purge_organisms();
    }

    pub fn reproduce(self: *SequentialPopulationEpochExecutor, opts: *Options, generation: usize, p: *Population) !void {
        std.debug.print("POPULATION: Start Sequential Reproduction Cycle >>>>>\n", .{});

        // Perform reproduction. Reproduction is done on a per-Species basis
        var offspring = std.ArrayList(*Organism).init(self.allocator);
        defer offspring.deinit();

        for (p.species.items) |sp| {
            var rep_offspring = try sp.reproduce(opts, generation, p, self.sorted_species);
            if (sp.id == self.best_species_id) {
                // store flag if best species reproduced - it will be used to determine if best species
                // produced offspring before died
                self.best_species_reproduced = true;
            }
            // store offspring
            try offspring.appendSliceAssumeCapacity(rep_offspring);
        }

        // sanity check - make sure that population size keep the same
        if (offspring.items.len != opts.pop_size) {
            std.debug.print("progeny size after reproduction cycle dimished, expected: {d}, but got: {d}\n", .{ opts.pop_size, offspring.items.len });
            return error.ReproductionPopSizeMismatch;
        }

        try p.speciate(opts, offspring.items);
        std.debug.print("POPULATION: >>>>> Reproduction Complete\n", .{});
    }

    pub fn finalize_reproduction(self: *SequentialPopulationEpochExecutor, _: *Options, pop: *Population) !void {
        // Destroy and remove the old generation from the organisms and species
        try pop.purge_old_generation();

        // Removes all empty Species and age ones that survive.
        // As this happens, create master organism list for the new generation.
        try pop.purge_or_age_species();

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
        self.sequential.deinit();
        self.allocator.destroy(self);
    }

    pub fn next_epoch(self: *ParallelPopulationEpochExecutor, opts: *Options, generation: usize, population: *Population) !void {
        self.sequential = try SequentialPopulationEpochExecutor.init(self.allocator);
        try self.sequential.prepare_for_reproduction(opts, generation, population);

        // Do parallel reproduction
        try self.reproduce(opts, generation, population);
        std.debug.print("POPULATION: >>>>> Epoch {d} complete\n", .{generation});
    }

    pub fn reproduce(self: *ParallelPopulationEpochExecutor, opts: *Options, generation: usize, pop: *Population) !void {
        std.debug.print("POPULATION: Start Parallel Reproduction Cycle >>>>>\n", .{});

        // Perform reproduction. Reproduction is done on a per-Species basis
        var sp_num = pop.species.items.len;
        var ctx = try WorkerCtx.init(self.allocator, sp_num);
        defer ctx.deinit();
        var pool: std.Thread.Pool = undefined;
        try pool.init(.{ .allocator = self.allocator });
        defer pool.deinit();

        var wait_group: WaitGroup = .{};
        for (pop.species.items) |species| {
            wait_group.start();
            try pool.spawn(workerFn, .{ ctx, &wait_group, species, opts, generation, pop, self.sequential.sorted_species.items });
        }
        wait_group.wait();

        // read reproduction results, instantiate progeny and speciate over population
        var babies = std.ArrayList(*Organism).init(self.allocator);
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
        try pop.speciate(opts, babies.items);

        std.debug.print("POPULATION: >>>>> Reproduction Complete", .{});
    }
};

pub const WorkerCtx = struct {
    allocator: std.heap.ThreadSafeAllocator,
    mu: std.Thread.Mutex = .{},
    res: []*ReproductionResult,

    pub fn init(allocator: std.mem.Allocator, count: usize) !*WorkerCtx {
        var self = try allocator.create(WorkerCtx);
        self.* = .{
            .allocator = std.heap.ThreadSafeAllocator{ .child_allocator = allocator },
            .res = try allocator.alloc(*ReproductionResult, count),
        };
        return self;
    }

    pub fn deinit(self: *WorkerCtx) void {
        for (self.res) |res| {
            if (res.babies != null) {
                self.allocator.free(res.babies.?);
            }
        }
        self.allocator.free(self.res);
        self.allocator.destroy(self);
    }
};

fn workerFn(ctx: *WorkerCtx, wg: *WaitGroup, species: *Species, opts: *Options, generation: usize, pop: *Population, sorted_species: []*Species, idx: usize) void {
    defer wg.finish();
    var res = try ReproductionResult.init(ctx.allocator.allocator(), species.id);
    res.babies = species.reproduce(opts, generation, pop, sorted_species) catch null;
    ctx.mu.lock();
    defer ctx.mu.unlock();
    ctx.res[idx] = res;
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
