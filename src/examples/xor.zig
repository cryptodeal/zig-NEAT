const std = @import("std");
const neat_population = @import("../genetics/population.zig");
const neat_organism = @import("../genetics/organism.zig");
const neat_trait = @import("../trait.zig");
const network_nnode = @import("../network/nnode.zig");
const exp_generation = @import("../experiment/generation.zig");
const neat_opts = @import("../opts.zig");
const neat_math = @import("../math/activations.zig");
const neat_gene = @import("../genetics/gene.zig");
const neat_genome = @import("../genetics/genome.zig");
const neat_experiment = @import("../experiment/experiment.zig");
const neat_exp_common = @import("../experiment/common.zig");
const neat_common = @import("../network/common.zig");

const Options = neat_opts.Options;
const GenerationEvaluator = neat_exp_common.GenerationEvaluator;
const Genome = neat_genome.Genome;
const Experiment = neat_experiment.Experiment;
const Gene = neat_gene.Gene;
const NodeActivationType = neat_math.NodeActivationType;
const NodeNeuronType = neat_common.NodeNeuronType;
const NNode = network_nnode.NNode;
const Trait = neat_trait.Trait;
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
        var input = in[count];
        organism.phenotype.?.load_sensors(&input);

        // Use depth to ensure full relaxation
        success = organism.phenotype.?.forward_steps(net_depth) catch {
            std.debug.print("Failed to activate network (failed @ call to `forward_steps`)\n", .{});
            return false;
        };

        out[count] = organism.phenotype.?.outputs[0].activation;

        // Flush network for subsequent use
        _ = try organism.phenotype.?.flush();
    }

    if (success) {
        // Mean Squared Error
        var error_sum: f64 = @fabs(out[0]) + @fabs(1 - out[1]) + @fabs(1 - out[2]) + @fabs(out[3]); // ideal == 0
        var target: f64 = 4 - error_sum;
        organism.fitness = std.math.pow(f64, 4 - error_sum, 2);
        organism.error_value = std.math.pow(f64, 4 - target, 2);
    } else {
        // The network is flawed (shouldn't happen) - flag as anomaly
        organism.error_value = 1.0;
        organism.fitness = 0.0;
    }

    if (organism.fitness > fitness_threshold) {
        organism.is_winner = true;
        std.debug.print(">>>> Output activations: {any}\n", .{out});
    } else {
        organism.is_winner = false;
    }

    return organism.is_winner;
}

fn eval(opts: *Options, pop: *Population, epoch: *Generation) !void {
    // Evaluate each organism on a test
    for (pop.organisms.items) |org| {
        var res = try org_eval(org);
        std.debug.print("\nGenome id: {d} ---- Organism fitness: {d}\n", .{ org.genotype.id, org.fitness });

        if (res and (epoch.champion == null or org.fitness > epoch.champion.?.fitness)) {
            epoch.solved = true;
            epoch.winner_nodes = org.genotype.nodes.len;
            epoch.winner_genes = @as(usize, @intCast(org.genotype.extrons()));
            epoch.winner_evals = opts.pop_size * epoch.id + @as(usize, @intCast(org.genotype.id));
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
        var org: *Organism = epoch.champion.?;
        std.debug.print("Winner organism fitness: {d}\n", .{org.fitness});

        var depth = try org.phenotype.?.max_activation_depth_fast(0);
        std.debug.print("Activation depth of the winner: {d}\n", .{depth});

        // TODO: write winner's genome to file (not implemented yet)
    }
}

pub fn main() !void {
    var allocator = std.heap.c_allocator;
    var opts: *Options = try allocator.create(Options);
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
        .mut_connect_sensors = 0.5,
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
        .epoch_executor_type = EpochExecutorType.EpochExecutorTypeSequential,
        .gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodFast,
    };

    // initialize Traits for Genome
    var traits = try allocator.alloc(*Trait, 3);
    for (traits, 0..) |_, i| {
        traits[i] = try Trait.init(allocator, 8);
        traits[i].id = @as(i64, @intCast(i)) + 1;
        traits[i].params[0] = 0.1 * @as(f64, @floatFromInt(i));
    }

    // initialize Nodes for Genome
    var nodes = try allocator.alloc(*NNode, 4);
    nodes[0] = try NNode.new_NNode(allocator, 1, NodeNeuronType.BiasNeuron);
    nodes[0].activation_type = NodeActivationType.NullActivation;
    // input Nodes
    nodes[1] = try NNode.new_NNode(allocator, 2, NodeNeuronType.InputNeuron);
    nodes[1].activation_type = NodeActivationType.NullActivation;
    nodes[2] = try NNode.new_NNode(allocator, 3, NodeNeuronType.InputNeuron);
    nodes[2].activation_type = NodeActivationType.NullActivation;
    // output Node
    nodes[3] = try NNode.new_NNode(allocator, 4, NodeNeuronType.OutputNeuron);
    nodes[3].activation_type = NodeActivationType.SigmoidSteepenedActivation;

    // initialize Genes for Genome
    var genes = try allocator.alloc(*Gene, 3);
    genes[0] = try Gene.init_with_trait(allocator, traits[0], 0.0, nodes[0], nodes[3], false, 1, 0);
    genes[1] = try Gene.init_with_trait(allocator, traits[0], 0.0, nodes[1], nodes[3], false, 1, 0);
    genes[2] = try Gene.init_with_trait(allocator, traits[0], 0.0, nodes[2], nodes[3], false, 1, 0);

    // initialize Genome
    var start_genome = try Genome.init(allocator, 1, traits, nodes, genes);
    defer start_genome.deinit();
    var experiment = try Experiment.init(allocator, 0);
    try experiment.trials.ensureTotalCapacityPrecise(opts.num_runs);
    defer experiment.deinit();

    const evaluator = GenerationEvaluator{ .generation_evaluate = eval };

    try experiment.execute(allocator, opts, start_genome, evaluator);

    var res = try experiment.avg_winner_statistics(allocator);
    defer res.deinit();
}

test "XOR" {
    var allocator = std.testing.allocator;
    var opts: *Options = try allocator.create(Options);
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
        .mut_connect_sensors = 0.5,
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
        .epoch_executor_type = EpochExecutorType.EpochExecutorTypeSequential,
        .gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodFast,
    };

    // initialize Traits for Genome
    var traits = try allocator.alloc(*Trait, 3);
    for (traits, 0..) |_, i| {
        traits[i] = try Trait.init(allocator, 8);
        traits[i].id = @as(i64, @intCast(i)) + 1;
        traits[i].params[0] = 0.1 * @as(f64, @floatFromInt(i));
    }

    // initialize Nodes for Genome
    var nodes = try allocator.alloc(*NNode, 4);
    nodes[0] = try NNode.new_NNode(allocator, 1, NodeNeuronType.BiasNeuron);
    nodes[0].activation_type = NodeActivationType.NullActivation;
    // input Nodes
    nodes[1] = try NNode.new_NNode(allocator, 2, NodeNeuronType.InputNeuron);
    nodes[1].activation_type = NodeActivationType.NullActivation;
    nodes[2] = try NNode.new_NNode(allocator, 3, NodeNeuronType.InputNeuron);
    nodes[2].activation_type = NodeActivationType.NullActivation;
    // output Node
    nodes[3] = try NNode.new_NNode(allocator, 4, NodeNeuronType.OutputNeuron);
    nodes[3].activation_type = NodeActivationType.SigmoidSteepenedActivation;

    // initialize Genes for Genome
    var genes = try allocator.alloc(*Gene, 3);
    genes[0] = try Gene.init_with_trait(allocator, traits[0], 0.0, nodes[0], nodes[3], false, 1, 0);
    genes[1] = try Gene.init_with_trait(allocator, traits[0], 0.0, nodes[1], nodes[3], false, 1, 0);
    genes[2] = try Gene.init_with_trait(allocator, traits[0], 0.0, nodes[2], nodes[3], false, 1, 0);

    // initialize Genome
    var start_genome = try Genome.init(allocator, 1, traits, nodes, genes);
    defer start_genome.deinit();
    var experiment = try Experiment.init(allocator, 0);
    try experiment.trials.ensureTotalCapacityPrecise(opts.num_runs);
    defer experiment.deinit();

    const evaluator = GenerationEvaluator{ .generation_evaluate = eval };

    try experiment.execute(allocator, opts, start_genome, evaluator);

    var res = try experiment.avg_winner_statistics(allocator);
    defer res.deinit();
}
