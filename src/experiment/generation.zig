const std = @import("std");
const neat_organism = @import("../genetics/organism.zig");
const neat_population = @import("../genetics/population.zig");
const floats = @import("floats.zig");

const fitnessComparison = neat_organism.fitnessComparison;
const Population = neat_population.Population;
const Organism = neat_organism.Organism;
const mean = floats.mean;

/// Generation represents execution results of one generation.
pub const Generation = struct {
    /// The generation Id for this epoch.
    id: usize,
    /// The time when epoch was evaluated.
    executed: std.time.Instant = undefined,
    /// The elapsed time between generation execution start and finish.
    duration: u64 = undefined,
    /// The best organism of the best species (probably successful solver if Solved flag set).
    champion: ?*Organism = null,
    /// The flag to indicate whether experiment was solved in this epoch.
    solved: bool = false,

    /// The list of the best organisms' fitness values per species in population.
    fitness: []f64 = undefined,
    /// The age of the best organisms' per species in population.
    age: []f64 = undefined,
    /// The list of the best organisms' complexities per species in population.
    complexity: []f64 = undefined,

    /// The number of species in population at the end of this epoch.
    diversity: usize = undefined,

    /// The number of evaluations done before winner (champion solver) found.
    winner_evals: usize = 0,
    /// The number of nodes in the genome of the winner (champion solver) or zero if not solved.
    winner_nodes: usize = 0,
    /// The numbers of genes (links) in the genome of the winner (champion solver) or zero if not solved.
    winner_genes: usize = 0,

    /// The Id of Trial this Generation was evaluated in.
    trial_id: usize,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new Generation.
    pub fn init(allocator: std.mem.Allocator, id: usize, trial_id: usize) !*Generation {
        var self: *Generation = try allocator.create(Generation);
        self.* = .{
            .id = id,
            .trial_id = trial_id,
            .allocator = allocator,
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *Generation) void {
        self.allocator.free(self.fitness);
        self.allocator.free(self.age);
        self.allocator.free(self.complexity);
        self.allocator.destroy(self);
    }

    /// Frees all associated memory if experiment fails with error.
    pub fn deinitEarly(self: *Generation) void {
        self.allocator.destroy(self);
    }

    /// Collects statistics about given population.
    pub fn fillPopulationStatistics(self: *Generation, pop: *Population) !void {
        var max_fitness: f64 = @as(f64, @floatFromInt(std.math.minInt(i64)));
        self.diversity = pop.species.items.len;
        // since these set Generation struct fields, alloc using Generation allocator
        self.age = try self.allocator.alloc(f64, self.diversity);
        self.complexity = try self.allocator.alloc(f64, self.diversity);
        self.fitness = try self.allocator.alloc(f64, self.diversity);
        for (pop.species.items, 0..) |curr_species, i| {
            self.age[i] = @as(f64, @floatFromInt(curr_species.age));
            // sort organisms from current species by fitness to have most fit first
            std.mem.sort(*Organism, curr_species.organisms.items, {}, fitnessComparison);
            std.mem.reverse(*Organism, curr_species.organisms.items);
            self.complexity[i] = @as(f64, @floatFromInt(curr_species.organisms.items[0].phenotype.?.complexity()));
            self.fitness[i] = curr_species.organisms.items[0].fitness;

            // finds the best organism in epoch if not solved
            if (!self.solved and curr_species.organisms.items[0].fitness > max_fitness) {
                max_fitness = curr_species.organisms.items[0].fitness;
                self.champion = curr_species.organisms.items[0];
            }
        }
    }

    /// Average the average fitness, age, and complexity among the best organisms
    /// of each species in the population at the end of this epoch.
    pub fn average(self: *Generation) GenerationAvg {
        return .{
            .fitness = mean(f64, self.fitness),
            .age = mean(f64, self.age),
            .complexity = mean(f64, self.complexity),
        };
    }
};

/// GenerationAvg represents average statistics of one generation.
pub const GenerationAvg = struct {
    fitness: f64,
    age: f64,
    complexity: f64,
};
