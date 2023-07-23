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
const Species = zig_neat.genetics.Species;
const Experiment = zig_neat.experiment.Experiment;
const Generation = zig_neat.experiment.Generation;
const NodeActivationType = zig_neat.math.NodeActivationType;
const Network = zig_neat.network.Network;
const species_org_sort = zig_neat.genetics.species_org_sort;

const thirty_six_degress: f64 = 36 * @as(f64, std.math.pi) / 180;

// maximal number of time steps for Markov experiment
const markov_max_steps: f64 = 100000;

// maximal number of time steps for Non-Markov long run
const non_markov_long_max_steps: f64 = 100000;

// maximal number of time steps for Non-Markov generalization run
const non_markov_generalization_max_steps: f64 = 1000;

var logger = NeatLogger{ .log_level = std.log.Level.info };

const ActionType = enum(u8) {
    ContinuousAction,
    DiscreteAction,
};

const Cart2PoleData = struct {
    // flag indicating whether to apply Markov evaluation variant
    markov: bool = false,
    // flag indicating whether to use continuous activation or discrete
    action_type: ActionType = ActionType.ContinuousAction,
};

const CartPole = struct {
    // flag indicating that we are executing Markov experiment setup (known velocities information)
    is_markov: bool = false,
    // flag that we are looking at the champion in Non-Markov experiment
    non_markov_long: bool = false,
    // flag that we are testing champion's generalization
    generalization_test: bool = false,
    // state of the system (x, ∆x/∆t, θ1, ∆θ1/∆t, θ2, ∆θ2/∆t)
    state: []f64,
    // number of balanced time steps passed for current organism evaluation
    balanced_time_steps: usize = 0,
    jiggle_step: []f64 = undefined,

    // Queues used for Gruau's fitness which damps oscillations
    cart_pos_sum: f64 = 0,
    cart_velocity_sum: f64 = 0,
    pole_pos_sum: f64 = 0,
    pole_velocity_sum: f64 = 0,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, markov: bool) !*CartPole {
        var self: *CartPole = try allocator.create(CartPole);
        self.* = .{
            .allocator = allocator,
            .is_markov = markov,
            .state = try allocator.alloc(f64, 6),
            .jiggle_step = try allocator.alloc(f64, 1000),
        };
        return self;
    }

    pub fn deinit(self: *CartPole) void {
        self.allocator.free(self.state);
        self.allocator.free(self.jiggle_step);
        self.allocator.destroy(self);
    }

    pub fn eval_net(self: *CartPole, net: *Network, action_type: ActionType) !f64 {
        var non_markov_max = non_markov_generalization_max_steps;
        if (self.non_markov_long) {
            non_markov_max = non_markov_long_max_steps;
        }
        self.reset_state();

        // max depth of the network to be activated
        var net_depth = try net.max_activation_depth_fast(0);
        if (net_depth == 0) {
            // disconnected - assign minimal fitness to not completely exclude organism from evolution
            // returning only fitness of 1 step
            if (self.is_markov) {
                return 1;
            } else {
                return 0.0001;
            }
        }

        if (self.is_markov) {
            var input = try self.allocator.alloc(f64, 7);
            defer self.allocator.free(input);

            var steps: f64 = 0;
            while (steps < markov_max_steps) : (steps += 1) {
                input[0] = (self.state[0] + 2.4) / 4.8;
                input[1] = (self.state[1] + 1) / 2;
                input[2] = (self.state[2] + thirty_six_degress) / (thirty_six_degress * 2); // 0.52
                input[3] = (self.state[3] + 1) / 2;
                input[4] = (self.state[4] + thirty_six_degress) / (thirty_six_degress * 2); // 0.52
                input[5] = (self.state[5] + 1) / 2;
                input[6] = 0.5;

                net.load_sensors(input);

                // activate the network based on the input
                var res = try net.forward_steps(net_depth);
                if (!res) {
                    // If it loops, exit returning only fitness of 1 step
                    logger.err("Failed to activate Network!", .{}, @src());
                    return 1;
                }

                var action = net.outputs[0].activation;
                if (action_type == ActionType.DiscreteAction) {
                    // make action values discrete
                    if (action < 0.5) {
                        action = 0;
                    } else {
                        action = 1;
                    }
                }
                try self.perform_action(action, @as(usize, @intFromFloat(steps)));
                if (self.outside_bounds()) break; // stop if failure
            }
            return steps;
        } else {
            var input = try self.allocator.alloc(f64, 4);
            defer self.allocator.free(input);

            var steps: f64 = 0;
            while (steps < non_markov_max) : (steps += 1) {
                input[0] = self.state[0] / 4.8;
                input[1] = self.state[2] / 0.52;
                input[2] = self.state[4] / 0.52;
                input[3] = 1;

                net.load_sensors(input);
                // activate the network based on the input
                var res = try net.forward_steps(net_depth);
                if (!res) {
                    // If it loops, exit returning only fitness of 1 step
                    return 0.0001;
                }

                var action = net.outputs[0].activation;
                if (action_type == ActionType.DiscreteAction) {
                    // make action values discrete
                    if (action < 0.5) {
                        action = 0;
                    } else {
                        action = 1;
                    }
                }
                try self.perform_action(action, @as(usize, @intFromFloat(steps)));
                if (self.outside_bounds()) break; // stop if failure
            }

            // If we are generalizing we just need to balance it for a while
            if (self.generalization_test) {
                return @as(f64, @floatFromInt(self.balanced_time_steps));
            }

            // Sum last 100
            var jiggle_total: f64 = 0;
            if (steps >= 100 and !self.non_markov_long) {
                // Adjust for array bounds and count
                var count: usize = @as(usize, @intFromFloat(steps - 100));
                while (count < @as(usize, @intFromFloat(steps))) : (count += 1) {
                    jiggle_total += self.jiggle_step[count];
                }
            }
            if (!self.non_markov_long) {
                var non_markov_fitness: f64 = undefined;
                if (self.balanced_time_steps >= 100) {
                    // F = 0.1f1 + 0.9f2
                    non_markov_fitness = 0.1 * @as(f64, @floatFromInt(self.balanced_time_steps)) / 1000 + 0.9 * 0.75 / jiggle_total;
                } else {
                    // F = t / 1000
                    non_markov_fitness = 0.1 * @as(f64, @floatFromInt(self.balanced_time_steps)) / 1000;
                    logger.debug("Balanced time steps: {d}, jiggle: {d} ***\n", .{ self.balanced_time_steps, jiggle_total }, @src());
                }
                return non_markov_fitness;
            } else {
                return steps;
            }
        }
    }

    fn perform_action(self: *CartPole, action: f64, step_num: usize) !void {
        const tau: f64 = 0.01; // ∆t = 0.01s

        // Apply action to the simulated cart-pole
        // Runge-Kutta 4th order integration method
        var dydx = try self.allocator.alloc(f64, 6);
        defer self.allocator.free(dydx);
        for (dydx) |*v| v.* = 0;
        var i: usize = 0;
        while (i < 2) : (i += 1) {
            dydx[0] = self.state[1];
            dydx[2] = self.state[3];
            dydx[4] = self.state[5];
            self.step(action, self.state, dydx);
            self.rk4(action, dydx, self.state, tau);
        }

        // Record this state
        self.cart_pos_sum += @fabs(self.state[0]);
        self.cart_velocity_sum += @fabs(self.state[1]);
        self.pole_pos_sum += @fabs(self.state[2]);
        self.pole_velocity_sum += @fabs(self.state[3]);

        if (step_num < 1000) {
            self.jiggle_step[step_num] = @fabs(self.state[0]) + @fabs(self.state[1]) + @fabs(self.state[2]) + @fabs(self.state[3]);
        }

        if (!self.outside_bounds()) {
            self.balanced_time_steps += 1;
        }
    }

    fn step(_: *CartPole, action: f64, st: []f64, derivs: []f64) void {
        const mup: f64 = 0.000002;
        const gravity: f64 = -9.8;
        const force_mag: f64 = 10; // [N]
        const mass_cart: f64 = 1; // [kg]

        const mass_pole1: f64 = 1; // [kg]
        const length1: f64 = 0.5; // [m] - actually half the first pole's length

        const mass_pole2: f64 = 0.1; // [kg]
        const length2: f64 = 0.05; // [m] - actually half the second pole's length

        var force = (action - 0.5) * force_mag * 2;
        var cos_theta1 = @cos(st[2]);
        var sin_theta1 = @sin(st[2]);
        var g_sin_theta1 = gravity * sin_theta1;
        var cos_theta2 = @cos(st[4]);
        var sin_theta2 = @sin(st[4]);
        var g_sin_theta2 = gravity * sin_theta2;

        var ml1 = length1 * mass_pole1;
        var ml2 = length2 * mass_pole2;
        var temp1 = mup * st[3] / ml1;
        var temp2 = mup * st[5] / ml2;
        var fi1 = (ml1 * st[3] * st[3] * sin_theta1) + (0.75 * mass_pole1 * cos_theta1 * (temp1 + g_sin_theta1));
        var fi2 = (ml2 * st[5] * st[5] * sin_theta2) + (0.75 * mass_pole2 * cos_theta2 * (temp2 + g_sin_theta2));
        var mi1 = mass_pole1 * (1 - (0.75 * cos_theta1 * cos_theta1));
        var mi2 = mass_pole2 * (1 - (0.75 * cos_theta2 * cos_theta2));

        derivs[1] = (force + fi1 + fi2) / (mi1 + mi2 + mass_cart);
        derivs[3] = -0.75 * (derivs[1] * cos_theta1 + g_sin_theta1 + temp1) / length1;
        derivs[5] = -0.75 * (derivs[1] * cos_theta2 + g_sin_theta2 + temp2) / length2;
    }

    fn rk4(self: *CartPole, f: f64, dydx: []f64, yout: []f64, tau: f64) void {
        var yt = [_]f64{0} ** 6;
        var dym = [_]f64{0} ** 6;
        var dyt = [_]f64{0} ** 6;
        var hh = tau * 0.5;
        var h6 = tau / 6;
        var i: usize = 0;
        while (i < 6) : (i += 1) {
            yt[i] = yout[i] + hh * dydx[i];
        }
        self.step(f, &yt, &dyt);

        dyt[0] = yt[1];
        dyt[2] = yt[3];
        dyt[4] = yt[5];
        i = 0;
        while (i < 6) : (i += 1) {
            yt[i] = yout[i] + hh * dyt[i];
        }
        self.step(f, &yt, &dym);

        dym[0] = yt[1];
        dym[2] = yt[3];
        dym[4] = yt[5];
        i = 0;
        while (i < 6) : (i += 1) {
            yt[i] = yout[i] + tau * dym[i];
            dym[i] += dyt[i];
        }
        self.step(f, &yt, &dyt);

        dyt[0] = yt[1];
        dyt[2] = yt[3];
        dyt[4] = yt[5];
        i = 0;
        while (i < 6) : (i += 1) {
            yout[i] = yout[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
        }
    }

    fn outside_bounds(self: *CartPole) bool {
        const failure_angle: f64 = thirty_six_degress;
        return self.state[0] < -2.4 or self.state[0] > 2.4 or self.state[2] < -failure_angle or self.state[2] > failure_angle or self.state[4] < -failure_angle or self.state[4] > failure_angle;
    }

    fn reset_state(self: *CartPole) void {
        if (self.is_markov) {
            // Clear all fitness records
            self.cart_pos_sum = 0;
            self.cart_velocity_sum = 0;
            self.pole_pos_sum = 0;
            self.pole_velocity_sum = 0;
            for (self.state) |*v| {
                v.* = 0;
            }
        } else if (!self.generalization_test) {
            // The long run non-markov test
            for (self.state, 0..) |_, i| {
                if (i == 2) {
                    self.state[i] = @as(f64, std.math.pi) / 180; // one_degree
                } else {
                    self.state[i] = 0;
                }
            }
        }
        self.balanced_time_steps = 0; // Always count # of balanced time steps
    }
};

const Cart2PoleGenerationEvaluator = struct {
    data: *Cart2PoleData,

    pub fn generation_evaluate(self: *Cart2PoleGenerationEvaluator, opts: *Options, pop: *Population, epoch: *Generation) !void {
        var cart_pole = try CartPole.init(pop.allocator, self.data.markov);

        for (pop.organisms.items) |org| {
            var winner = try self.org_eval(org, cart_pole);
            if (winner and (epoch.champion == null or org.fitness > epoch.champion.?.fitness)) {
                // This will be winner in Markov case
                epoch.solved = true;
                epoch.winner_nodes = org.genotype.nodes.len;
                epoch.winner_genes = @as(usize, @intCast(org.genotype.extrons()));
                epoch.winner_evals = opts.pop_size * epoch.id + @as(usize, @intCast(org.genotype.id));
                epoch.champion = org;
                org.is_winner = true;
            }
        }

        // Check for winner in Non-Markov case
        if (!self.data.markov) {
            // The best individual (i.e. the one with the highest fitness value) of every generation is tested for
            // its ability to balance the system for a longer time period. If a potential solution passes this test
            // by keeping the system balanced for 100’000 time steps, the so called generalization score(GS) of this
            // particular individual is calculated. This score measures the potential of a controller to balance the
            // system starting from different initial conditions. It's calculated with a series of experiments, running
            // over 1000 time steps, starting from 625 different initial conditions.
            // The initial conditions are chosen by assigning each value of the set Ω = [0.05 0.25 0.5 0.75 0.95] to
            // each of the states x, ∆x/∆t, θ1 and ∆θ1/∆t, scaled to the range of the variables.The short pole angle θ2
            // and its angular velocity ∆θ2/∆t are set to zero. The GS is then defined as the number of successful runs
            // from the 625 initial conditions and an individual is defined as a solution if it reaches a generalization
            // score of 200 or more.

            // Sort the species by max organism fitness in descending order - the highest fitness first
            var sorted_species: []*Species = try pop.allocator.alloc(*Species, pop.species.items.len);
            defer pop.allocator.free(sorted_species);
            @memcpy(sorted_species, pop.species.items);
            std.mem.sort(*Species, sorted_species, {}, species_org_sort);
            std.mem.reverse(*Species, sorted_species);

            // First update what is checked and unchecked
            var curr_species: ?*Species = undefined;
            for (sorted_species, 0..) |_, i| {
                curr_species = sorted_species[i];
                var max = curr_species.?.compute_max_and_avg_fitness();
                if (max.max > curr_species.?.max_fitness_ever) {
                    curr_species.?.is_checked = false;
                }
            }

            // Now find first (most fit) species that is unchecked
            curr_species = null;
            for (sorted_species, 0..) |_, i| {
                curr_species = sorted_species[i];
                if (!curr_species.?.is_checked) {
                    break;
                }
            }

            if (curr_species == null) {
                curr_species = sorted_species[0];
            }

            // Remember it was checked
            curr_species.?.is_checked = true;

            // the organism champion
            var champion = curr_species.?.find_champion();
            var champion_fitness = champion.?.fitness;

            // Now check to make sure the champion can do 100,000 evaluations
            cart_pole.non_markov_long = true;
            cart_pole.generalization_test = false;

            var long_run_passed = try self.org_eval(champion.?, cart_pole);
            if (long_run_passed) {
                // the champion passed non-Markov long test, start generalization
                cart_pole.non_markov_long = false;
                cart_pole.generalization_test = true;

                // Given that the champion passed long run test, now run it on generalization tests running
                // over 1000 time steps, starting from 625 different initial conditions

                var state_vals = [_]f64{ 0.05, 0.25, 0.5, 0.75, 0.95 };
                var generalization_score: usize = 0;
                var s0c: usize = 0;
                var s1c: usize = 0;
                var s2c: usize = 0;
                var s3c: usize = 0;
                while (s0c < 5) : (s0c += 1) {
                    while (s1c < 5) : (s1c += 1) {
                        while (s1c < 5) : (s1c += 1) {
                            while (s2c < 5) : (s2c += 1) {
                                while (s3c < 5) : (s3c += 1) {
                                    cart_pole.state[0] = state_vals[s0c] * 4.32 - 2.16;
                                    cart_pole.state[1] = state_vals[s1c] * 2.70 - 1.35;
                                    cart_pole.state[2] = state_vals[s2c] * 0.12566304 - 0.06283152; // 0.06283152 = 3.6 degrees
                                    cart_pole.state[3] = state_vals[s3c] * 0.30019504 - 0.15009752; // 0.15009752 = 8.6 degrees
                                    // The short pole angle and its angular velocity are set to zero.
                                    cart_pole.state[4] = 0;
                                    cart_pole.state[5] = 0;

                                    // The champion needs to be flushed here because it may have
                                    // leftover activation from its last test run that could affect
                                    // its recurrent memory
                                    _ = try champion.?.phenotype.?.flush();
                                    var generalized = try self.org_eval(champion.?, cart_pole);
                                    if (generalized) {
                                        generalization_score += 1;
                                        logger.debug("x: {d}, xv: {d}, t1: {d}, t2: {d}, angle: {d}", .{ cart_pole.state[0], cart_pole.state[1], cart_pole.state[2], cart_pole.state[4], thirty_six_degress }, @src());
                                    }
                                }
                            }
                        }
                    }
                }

                if (generalization_score >= 200) {
                    // The generalization test winner
                    logger.info("The non-Markov champion found! (Generalization Score = {d})", .{generalization_score}, @src());

                    champion.?.fitness = @as(f64, @floatFromInt(generalization_score));
                    champion.?.is_winner = true;
                    epoch.solved = true;
                    epoch.winner_nodes = champion.?.genotype.nodes.len;
                    epoch.winner_genes = @as(usize, @intCast(champion.?.genotype.extrons()));
                    epoch.winner_evals = opts.pop_size * epoch.id + @as(usize, @intCast(champion.?.genotype.id));
                    epoch.champion = champion.?;
                } else {
                    logger.info("The non-Markov champion missed the 100'000 run test", .{}, @src());
                    champion.?.fitness = champion_fitness; // Restore champ's fitness
                    champion.?.is_winner = false;
                }
            }
        }

        // Fill statistics about current epoch
        try epoch.fill_population_statistics(pop);

        // TODO: Only print to file every print_every generation
        if (epoch.solved) {
            // print winner organism
            var org: *Organism = epoch.champion.?;
            var depth = try org.phenotype.?.max_activation_depth_fast(0);
            std.debug.print("Activation depth of the winner: {d}\n", .{depth});

            // TODO: write winner's genome to file (not implemented yet)
        }
    }

    fn org_eval(self: *Cart2PoleGenerationEvaluator, org: *Organism, cart_pole: *CartPole) !bool {
        var winner = false;
        // Try to balance a pole now
        org.fitness = try cart_pole.eval_net(org.phenotype.?, self.data.action_type);

        logger.debug("Organism {d}\tfitness: {d}", .{ org.genotype.id, org.fitness }, @src());

        // DEBUG CHECK if organism is damaged
        if (!(cart_pole.non_markov_long and cart_pole.generalization_test) and org.check_champion_child_damaged()) {
            logger.warn("ORGANISM DEGRADED:\n{any}", .{org.genotype}, @src());
        }

        // Decide if it's a winner, in Markov Case
        if (cart_pole.is_markov) {
            if (org.fitness >= markov_max_steps) {
                winner = true;
                org.fitness = 1;
                org.error_value = 0;
            } else {
                // use linear scale
                org.error_value = (markov_max_steps - org.fitness) / markov_max_steps;
                org.fitness = 1 - org.error_value;
            }
        } else if (cart_pole.non_markov_long) {
            // if doing the long test non-markov
            if (org.fitness >= non_markov_long_max_steps) {
                winner = true;
            }
        } else if (cart_pole.generalization_test) {
            if (org.fitness >= non_markov_generalization_max_steps) {
                winner = true;
            }
        }
        return winner;
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
        .weight_mut_power = 2.5,
        .disjoint_coeff = 1,
        .excess_coeff = 1,
        .mut_diff_coeff = 3,
        .compat_threshold = 3,
        .age_significance = 1,
        .survival_thresh = 0.2,
        .mut_only_prob = 0.25,
        .mut_random_trait_prob = 0.1,
        .mut_link_trait_prob = 0.1,
        .mut_node_trait_prob = 0.1,
        .mut_link_weights_prob = 0.9,
        .mut_toggle_enable_prob = 0.1,
        .mut_gene_reenable_prob = 0.05,
        .mut_add_node_prob = 0.3,
        .mut_add_link_prob = 0.5,
        .mut_connect_sensors = 0.5,
        .interspecies_mate_rate = 0.01,
        .mate_multipoint_prob = 0.6,
        .mate_multipoint_avg_prob = 0.4,
        .mate_singlepoint_prob = 0,
        .mate_only_prob = 0.2,
        .recur_only_prob = 0.1,
        .pop_size = 1000,
        .dropoff_age = 15,
        .new_link_tries = 20,
        .print_every = 30,
        .babies_stolen = 0,
        .num_runs = 10,
        .num_generations = 100,
        .node_activators = node_activators,
        .node_activators_prob = node_activators_prob,
        .epoch_executor_type = EpochExecutorType.EpochExecutorTypeSequential,
        .gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodLinear,
    };

    // initialize Genome
    var start_genome = try Genome.read_from_file(allocator, "data/pole2_markov_startgenes");
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

    var ctx = Cart2PoleData{
        .markov = true,
        .action_type = ActionType.ContinuousAction,
    };

    var cart2pole_eval = Cart2PoleGenerationEvaluator{
        .data = &ctx,
    };

    const evaluator = GenerationEvaluator.init(&cart2pole_eval);

    try experiment.execute(allocator, rand, opts, start_genome, evaluator, null);

    // var res = experiment.avg_winner_statistics();
}
