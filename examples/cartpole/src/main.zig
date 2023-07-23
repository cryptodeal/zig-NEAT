const std = @import("std");
const zig_neat = @import("zigNEAT");

const NeatLogger = zig_neat.NeatLogger;
const Population = zig_neat.genetics.Population;
const Organism = zig_neat.genetics.Organism;
const Options = zig_neat.Options;
const EpochExecutorType = zig_neat.EpochExecutorType;
const GenomeCompatibilityMethod = zig_neat.GenomeCompatibilityMethod;
const GenerationEvaluator = zig_neat.experiment.GenerationEvaluator;
const Genome = zig_neat.genetics.Genome;
const Experiment = zig_neat.experiment.Experiment;
const Generation = zig_neat.experiment.Generation;
const NodeActivationType = zig_neat.math.NodeActivationType;
const Network = zig_neat.network.Network;

var logger = NeatLogger{ .log_level = std.log.Level.info };

const twelve_degrees: f64 = 12 * @as(f64, std.math.pi) / 180;

const CartPoleGenerationEvaluator = struct {
    allocator: std.mem.Allocator,
    // The flag to indicate if cart emulator should be started from random position
    random_start: bool = true,
    // The number of emulation steps to be done balancing pole to win
    win_balance_steps: usize = 500000,

    pub fn generation_evaluate(self: *CartPoleGenerationEvaluator, opts: *Options, pop: *Population, epoch: *Generation) !void {
        // evaluate each organism on a test
        for (pop.organisms.items) |org| {
            var res = try self.org_eval(org);
            logger.debug("Organism: {d}\tComplexity: {d}\tFitness: {d}", .{ org.genotype.id, org.phenotype.?.complexity(), org.fitness }, @src());

            if (res and (epoch.champion == null or org.fitness > epoch.champion.?.fitness)) {
                epoch.solved = true;
                epoch.winner_nodes = org.genotype.nodes.len;
                epoch.winner_genes = @as(usize, @intCast(org.genotype.extrons()));
                epoch.winner_evals = opts.pop_size * epoch.id + @as(usize, @intCast(org.genotype.id));
                epoch.champion = org;
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

    fn org_eval(self: *CartPoleGenerationEvaluator, org: *Organism) !bool {
        // Try to balance a pole now
        var fitness = self.run_cart(self.allocator, org.phenotype.?) catch return false;
        org.fitness = @as(f64, @floatFromInt(fitness));

        // Decide if it's a winner
        if (org.fitness >= @as(f64, @floatFromInt(self.win_balance_steps))) {
            org.is_winner = true;
        }

        // adjust fitness to be in range [0;1]
        if (org.is_winner) {
            org.fitness = 1;
            org.error_value = 0;
        } else if (org.fitness == 0) {
            org.error_value = 1.0;
        } else {
            // we use logarithmic scale because most cart runs fail to early within ~100 steps, but
            // we test against 500'000 balancing steps
            var log_steps = @log(@as(f64, @floatFromInt(self.win_balance_steps)));
            org.error_value = (log_steps - @log(org.fitness)) / log_steps;
            org.fitness = 1 - org.error_value;
        }

        return org.is_winner;
    }

    fn run_cart(self: *CartPoleGenerationEvaluator, allocator: std.mem.Allocator, net: *Network) !usize {
        var x: f64 = 0; // cart position, meters
        var x_dot: f64 = 0; // cart velocity
        var theta: f64 = 0; // pole angle, radians
        var theta_dot: f64 = 0; // pole angular velocity

        var prng = std.rand.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.os.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });
        const rand = prng.random();

        if (self.random_start) {
            // random starting position
            x = @as(f64, @floatFromInt(@mod(@as(i32, @intCast(rand.int(i31))), 4800))) / 1000 - 2.4;
            x_dot = @as(f64, @floatFromInt(@mod(@as(i32, @intCast(rand.int(i31))), 2000))) / 1000 - 1;
            theta = @as(f64, @floatFromInt(@mod(@as(i32, @intCast(rand.int(i31))), 400))) / 1000 - 0.2;
            theta_dot = @as(f64, @floatFromInt(@mod(@as(i32, @intCast(rand.int(i31))), 3000))) / 1000 - 1.5;
        }

        var net_depth = try net.max_activation_depth_fast(0);
        if (net_depth == 0) {
            // possibly disconnected - return minimal fitness score
            logger.err("Failed to estimate maximal depth of the network with loop.\nUsing default depth: {d}", .{net_depth}, @src());
            return 1;
        }

        var in = try allocator.alloc(f64, 5);
        defer allocator.free(in);
        var steps: usize = 0;
        while (steps < self.win_balance_steps) : (steps += 1) {
            // setup the input layer based on the four inputs
            in[0] = 1; // bias
            in[1] = (x + 2.4) / 4.8;
            in[2] = (x_dot + 0.75) / 1.5;
            in[3] = (theta + twelve_degrees) / 0.41;
            in[4] = (theta_dot + 1) / 2;
            net.load_sensors(in);

            // activate the network based on the input
            var res = try net.forward_steps(net_depth);
            if (!res) {
                // if it loops, exit returning only fitness of 1 step
                logger.err("Failed to activate Network!", .{}, @src());
                return 1;
            }

            // decide which way to push via which output unit is greater
            var action: u8 = 1;
            if (net.outputs[0].activation > net.outputs[1].activation) {
                action = 0;
            }

            // apply action to the simulated cart_pole
            self.simulate_action(action, &x, &x_dot, &theta, &theta_dot);

            // Check for failure.  If so, return steps
            if (x < -2.4 or x > 2.4 or theta < -twelve_degrees or theta > twelve_degrees) {
                return steps;
            }
        }
        return steps;
    }

    // simulate_action was taken directly from the pole simulator written by Richard Sutton and Charles Anderson.
    // This simulator uses normalized, continuous inputs instead of discretizing the input space.
    //   - Takes an action (0 or 1) and the current values of the
    //   - four state variables and updates their values by estimating the state
    //   - TAU seconds later.
    fn simulate_action(_: *CartPoleGenerationEvaluator, action: u8, x: *f64, x_dot: *f64, theta: *f64, theta_dot: *f64) void {
        var x_copy = x.*;
        var x_dot_copy = x_dot.*;
        var theta_copy = theta.*;
        var theta_dot_copy = theta_dot.*;
        // The cart pole configuration values
        const gravity: f64 = 9.8;
        const mass_cart: f64 = 1;
        const mass_pole: f64 = 0.5;
        const total_mass: f64 = mass_cart + mass_pole;
        const length: f64 = 0.5; // actually half the pole's length
        const pole_mass_length: f64 = mass_pole * length;
        const force_mag: f64 = 10;
        const tau = 0.02; // seconds between state updates
        const four_thirds: f64 = 1.3333333333333;

        var force = -force_mag;
        if (action > 0) {
            force = force_mag;
        }

        var cos_theta = @cos(theta_copy);
        var sin_theta = @sin(theta_copy);

        var temp = (force + pole_mass_length * theta_dot_copy * theta_dot_copy * sin_theta) / total_mass;

        var theta_acc = (gravity * sin_theta - cos_theta * temp) / (length * (four_thirds - mass_pole * cos_theta * cos_theta / total_mass));

        var x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;

        // update the four state variables, using Euler's method
        x.* = x_copy + tau * x_dot_copy;
        x_dot.* = x_dot_copy + tau * x_acc;
        theta.* = theta_copy + tau * theta_dot_copy;
        theta_dot.* = theta_dot_copy + tau * theta_acc;
    }
};

pub fn main() !void {
    var allocator = std.heap.c_allocator;
    var opts: *Options = try allocator.create(Options);
    defer allocator.destroy(opts);
    var node_activators = try allocator.alloc(NodeActivationType, 1);
    defer allocator.free(node_activators);
    node_activators[0] = NodeActivationType.SigmoidSteepenedActivation;
    var node_activators_prob = try allocator.alloc(f64, 1);
    defer allocator.free(node_activators_prob);
    node_activators_prob[0] = 1.0;
    opts.* = .{
        .trait_param_mut_prob = 0.5,
        .trait_mut_power = 1,
        .weight_mut_power = 1.8,
        .disjoint_coeff = 1,
        .excess_coeff = 1,
        .mut_diff_coeff = 3,
        .compat_threshold = 4,
        .age_significance = 1,
        .survival_thresh = 0.4,
        .mut_only_prob = 0.25,
        .mut_random_trait_prob = 0.1,
        .mut_link_trait_prob = 0.1,
        .mut_node_trait_prob = 0.1,
        .mut_link_weights_prob = 0.8,
        .mut_toggle_enable_prob = 0.1,
        .mut_gene_reenable_prob = 0.05,
        .mut_add_node_prob = 0.01,
        .mut_add_link_prob = 0.3,
        .mut_connect_sensors = 0.5,
        .interspecies_mate_rate = 0.001,
        .mate_multipoint_prob = 0.6,
        .mate_multipoint_avg_prob = 0.4,
        .mate_singlepoint_prob = 0,
        .mate_only_prob = 0.2,
        .recur_only_prob = 0.2,
        .pop_size = 1000,
        .dropoff_age = 15,
        .new_link_tries = 20,
        .print_every = 60,
        .babies_stolen = 0,
        .num_runs = 100,
        .num_generations = 100,
        .node_activators = node_activators,
        .node_activators_prob = node_activators_prob,
        .epoch_executor_type = EpochExecutorType.EpochExecutorTypeSequential,
        .gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodLinear,
    };

    // initialize Genome
    var start_genome = try Genome.read_from_file(allocator, "data/pole1startgenes");
    defer start_genome.deinit();

    var experiment = try Experiment.init(allocator, 0);
    defer experiment.deinit();
    try experiment.trials.ensureTotalCapacityPrecise(opts.num_runs);

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    var cartpole_evaluator = CartPoleGenerationEvaluator{ .allocator = allocator };

    const evaluator = GenerationEvaluator.init(&cartpole_evaluator);

    try experiment.execute(allocator, rand, opts, start_genome, evaluator, null);

    // var res = experiment.avg_winner_statistics();
}
