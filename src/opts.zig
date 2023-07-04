const std = @import("std");
const math = @import("math/activations.zig");
const m = @import("math/math.zig");
const NeatLogger = @import("log.zig").NeatLogger;

pub const EpochExecutorType = enum {
    EpochExecutorTypeSequential,
    EpochExecutorTypeParallel,
};

pub const GenomeCompatibilityMethod = enum {
    GenomeCompatibilityMethodLinear,
    GenomeCompatibilityMethodFast,
};

pub const logger = &NeatLogger{ .log_level = std.log.Level.debug };

pub const Options = struct {
    // probability of mutating single trait param
    trait_param_mut_prob: f64 = 0,
    // power of mutation on single trait param
    trait_mut_power: f64 = 0,
    // power of link weight mutation
    weight_mut_power: f64 = 0,

    disjoint_coeff: f64 = 0,
    excess_coeff: f64 = 0,
    mut_diff_coeff: f64 = 0,

    // globabl val representing compatability threshold
    // under which 2 Genomes are considered same species
    compat_threshold: f64 = 0,

    // globals used in the epoch cycle; mating, reproduction, etc...

    // how much should age matter? gives fitness boost
    // young age (niching). 1.0 = no age discrimination
    age_significance: f64 = 0,
    // pct of avg fitness for survival, how many get to
    // reproduce based on survival_thres * pop_size
    survival_thresh: f64 = 0,

    // probability of non-mating reproduction
    mut_only_prob: f64 = 0,
    mut_random_trait_prob: f64 = 0,
    mut_link_trait_prob: f64 = 0,
    mut_node_trait_prob: f64 = 0,
    mut_link_weights_prob: f64 = 0,
    mut_toggle_enable_prob: f64 = 0,
    mut_gene_reenable_prob: f64 = 0,
    mut_add_node_prob: f64 = 0,
    mut_add_link_prob: f64 = 0,
    // probability of mutation involving disconnected inputs cxn
    mut_connect_sensors: f64 = 0,

    // probabilities of a mate being outside species
    interspecies_mate_rate: f64 = 0,
    mate_multipoint_prob: f64 = 0,
    mate_multipoint_avg_prob: f64 = 0,
    mate_singlepoint_prob: f64 = 0,

    // probability of mating without mutation
    mate_only_prob: f64 = 0,
    // probability of forcing selection of ONLY links that are naturally recurrent
    recur_only_prob: f64 = 0,

    // size of population
    pop_size: usize = 0,
    // age when species starts to be penalized
    dropoff_age: i64 = 0,
    // number of times mutate_add_link will attempt to open new link
    new_link_tries: usize = 0,

    // write population to file every n generations
    print_every: usize = 0,

    // number of "babies" stolen off to champions
    babies_stolen: usize = 0,

    // number of runs to avg over in an experiment
    num_runs: usize = 0,

    // number of epochs (generations) to execute training
    num_generations: usize = 0,

    // specifies the epoch's exectuor type to apply
    epoch_executor_type: EpochExecutorType = undefined,
    // specifies the method used for genome compatability
    // test (linear, fast - better for large genomes)
    gen_compat_method: GenomeCompatibilityMethod = undefined,

    // neuron nodes activation function list to choose from
    node_activators: []math.NodeActivationType = undefined,
    // probabilities of selection of the specific node activator fn
    node_activators_prob: []f64 = undefined,

    // list of supported node activation with probability of each one
    node_activators_with_probs: [][]const u8 = undefined,

    // the log output details level
    log_level: []const u8 = "info",

    // allocator used internally
    allocator: std.mem.Allocator = undefined,

    pub fn random_node_activation_type(self: *Options) !math.NodeActivationType {
        // quick check for the most cases
        if (self.node_activators.len == 1) {
            return self.node_activators[0];
        }
        // find next random
        var idx = @as(usize, @intCast(m.single_roulette_throw(self.node_activators_prob)));
        if (idx < 0 or idx == self.node_activators.len) {
            std.debug.print("unexpected error when trying to find random node activator, activator index: {d}\n", .{idx});
            return error.NodeActivationTypeInvalid;
        }
        return self.node_activators[idx];
    }

    pub fn init_node_activators(self: *Options, allocator: std.mem.Allocator) !void {
        if (self.node_activators_with_probs.len == 0) {
            self.node_activators = try allocator.alloc(math.NodeActivationType, 1);
            self.node_activators[0] = math.NodeActivationType.SigmoidSteepenedActivation;
            return;
        }
        // create activators
        var act_fns = self.node_activators_with_probs;
        self.node_activators = try allocator.alloc(math.NodeActivationType, act_fns.len);
        self.node_activators_prob = try allocator.alloc(f64, act_fns.len);
        for (act_fns, 0..) |line, i| {
            var field_iterator = std.mem.split(u8, line, " ");
            self.node_activators[i] = math.node_activation_type_from_str(field_iterator.first());
            self.node_activators_prob[i] = try std.fmt.parseFloat(f64, field_iterator.rest());
        }
    }

    // TODO: maybe??? pub fn validate(self: *Options) !void {}

    pub fn load_json_opts(allocator: std.mem.Allocator, path: []const u8) !*Options {
        const data: []const u8 = try std.fs.cwd().readFileAlloc(allocator, path, 512);
        defer allocator.free(data);

        var self: *Options = try std.json.parseFromSlice(Options, allocator, data, .{});
        self.allocator = allocator;

        logger = try NeatLogger.init(allocator, self.log_level);

        // read node activators
        try self.init_node_activators(allocator);

        return &self;
    }
};
