const std = @import("std");
const neat_population = @import("../genetics/population.zig");
const neat_organism = @import("../genetics/organism.zig");
const exp_generation = @import("../experiment/generation.zig");
const neat_opts = @import("../opts.zig");

const Options = neat_opts.Options;
const EpochExecutorType = neat_opts.EpochExecutorType;
const GenomeCompatibilityMethod = neat_opts.GenomeCompatibilityMethod;
const Generation = exp_generation.Generation;
const Population = neat_population.Population;
const Organism = neat_organism.Organism;

pub const fitness_threshold: f64 = 15.5;

fn org_eval(organism: *Organism) !bool {
    // The four possible input combinations to xor
    // The first number is for biasing
    var in = [4][3]f64{
        [3]f64{ 1.0, 0.0, 0.0 },
        [3]f64{ 1.0, 0.0, 1.0 },
        [3]f64{ 1.0, 1.0, 0.0 },
        [3]f64{ 1.0, 1.0, 1.0 },
    };

    // The max depth of the network to be activated
    var net_depth = try organism.phenotype.?.max_activation_depth_fast(0);
    if (net_depth == 0) {
        std.debug.print("Network depth: {d} for organism: {d}\n", .{ net_depth, organism.genotype.id });
        return false;
    }

    // Check for successful activation
    var success = false;
    // The four outputs
    var out: [4]f64 = undefined;

    // Load and activate the network on each input
    var count: usize = 0;
    while (count < 4) : (count += 1) {
        organism.phenotype.?.load_sensors(in[count]);

        // Use depth to ensure full relaxation
        success = organism.phenotype.?.forward_steps(net_depth) catch {
            std.debug.print("Failed to activate network (failed @ call to `forward_steps`)\n", .{});
            return false;
        };

        out[count] = organism.phenotype.?.outputs[0].activation;

        // Flush network for subsequent use
        organism.phenotype.?.flush();
    }

    if (success) {
        // Mean Squared Error
        var error_sum: f64 = @fabs(out[0]) + @fabs(1.0 - out[1]) + @fabs(1.0 - out[2]) + @fabs(out[3]); // ideal == 0
        var target: f64 = 4.0 - error_sum;
        organism.fitness = std.math.pow(f64, 4.0 - error_sum, 2.0);
        organism.error_value = std.math.pow(f64, 4.0 - target, 2.0);
    } else {
        // The network is flawed (shouldn't happen) - flag as anomaly
        organism.error_value = 1.0;
        organism.fitness = 0.0;
    }

    if (organism.fitness > fitness_threshold) {
        organism.is_winner = true;
        std.debug.print(">>>> Output activations: {any}\n", out);
    } else {
        organism.is_winner = false;
    }

    return organism.is_winner;
}

fn eval(opts: *Options, pop: *Population, epoch: *Generation) !void {
    // Evaluate each organism on a test
    for (pop.organisms.items) |org| {
        var res = try org_eval(org);

        if (res and (epoch.champion == null or org.fitness > epoch.champion.?.fitness)) {
            epoch.solved = true;
            epoch.winner_nodes = org.genotype.nodes.len;
            epoch.winner_genes = @intCast(usize, org.genotype.extrons());
            epoch.winner_evals = opts.pop_size * epoch.id + @intCast(usize, org.genotype.id);
            epoch.champion = org;
            // TODO: add functionality to write genome as JSON
            //if (epoch.winner_nodes == 5) {
            // You could dump out optimal genomes here if desired
            //}
        }
    }

    // Fill statistics about current epoch
    try epoch.fill_population_statistics(pop);

    // TODO: Only print to file every print_every generation

    if (epoch.solved) {
        // print winner organism
        var org: *Organism = epoch.champion;

        var depth = try org.phenotype.?.max_activation_depth_fast(0);
        std.debug.print("Activation depth of the winner: {d}\n", .{depth});

        // TODO: write winner's genome to file (not implemented yet)
    }
}

test "XOR" {
    var allocator = std.testing.allocator;
    var opts: *Options = allocator.create(Options);
    defer allocator.destroy(opts);
    opts.* = .{
        .trait_param_mut_prob = 0.5,
        .trait_mut_power = 1.0,
        .weight_mut_power = 2.5,
        .disjoint_coeff = 1.0,
        .excess_coeff = 1.0,
        .mut_diff_coeff = 0.4,
        .compat_threshold = 3.0,
        .age_significance = 1.0,
        .survival_thresh = 0.2,
        .mut_only_prob = 0.25,
        .mut_random_trait_prob = 0.1,
        .mut_link_trait_prob = 0.1,
        .mut_node_trait_prob = 0.1,
        .mut_link_weights_prob = 0.9,
        .mut_toggle_enable_prob = 0.0,
        .mut_gene_reenable_prob = 0.0,
        .mut_add_node_prob = 0.03,
        .mut_add_link_prob = 0.08,
        .mut_connect_sensor = 0.5,
        .interspecies_mate_rate = 0.0010,
        .mate_multipoint_prob = 0.3,
        .mate_multipoint_avg_prob = 0.3,
        .mate_singlepoint_prob = 0.3,
        .mate_only_prob = 0.2,
        .recur_only_prob = 0.0,
        .pop_size = 200,
        .dropoff_age = 50,
        .new_link_tries = 50,
        .print_every = 10,
        .babies_stolen = 0,
        .num_runs = 100,
        .num_generations = 100,
        .epoch_executor = EpochExecutorType.EpochExecutorTypeSequential,
        .gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodLinear,
    };
}
