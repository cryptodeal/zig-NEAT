const std = @import("std");
const neat_genome = @import("genome.zig");
const neat_population = @import("population.zig");
const neat_options = @import("../opts.zig");
const neat_pop_executor = @import("population_epoch.zig");

const Options = neat_options.Options;
const Population = neat_population.Population;
const Genome = neat_genome.Genome;
const parallel_executor_next_epoch = neat_pop_executor.parallel_executor_next_epoch;

// TODO: fix - test leaks memory and fails
test "ParallelPopulationEpochExecutor next epoch" {
    var allocator = std.testing.allocator;

    var in: i64 = 3;
    var out: i64 = 2;
    var nmax: i64 = 15;
    var n: i64 = 3;
    var link_prob: f64 = 0.8;

    // configuration
    var opts = Options{
        .compat_threshold = 0.5,
        .dropoff_age = 1,
        .pop_size = 30,
        .babies_stolen = 10,
        .recur_only_prob = 0.2,
    };

    var gen = try Genome.init_rand(allocator, 1, in, out, n, nmax, false, link_prob);
    defer gen.deinit();
    var pop = try Population.init(allocator, gen, &opts);
    defer pop.deinit();

    var res = try parallel_executor_next_epoch(allocator, pop, &opts);
    try std.testing.expect(res);
}
