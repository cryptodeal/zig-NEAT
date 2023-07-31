const std = @import("std");
const math = @import("math/activations.zig");
const m = @import("math/math.zig");
const NeatLogger = @import("log.zig").NeatLogger;
const json = @import("json");
const utils = @import("utils/utils.zig");
const read_file = utils.read_file;
const get_writable_file = utils.get_writable_file;

pub const EpochExecutorType = enum {
    EpochExecutorTypeSequential,
    EpochExecutorTypeParallel,
};

pub const GenomeCompatibilityMethod = enum {
    GenomeCompatibilityMethodLinear,
    GenomeCompatibilityMethodFast,
};

pub const logger = &NeatLogger{ .log_level = std.log.Level.info };

const OptionsJSON = struct {
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

    // the log output details level
    log_level: std.log.Level = .info,

    // neuron nodes activation function list to choose from
    node_activators: []math.NodeActivationType = undefined,
    // probabilities of selection of the specific node activator fn
    node_activators_prob: []f64 = undefined,
};

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

    // the log output details level
    log_level: std.log.Level = .info,

    // allocator used internally
    allocator: std.mem.Allocator = undefined,

    pub fn jsonify(self: *Options) OptionsJSON {
        return .{
            .trait_param_mut_prob = self.trait_param_mut_prob,
            .trait_mut_power = self.trait_mut_power,
            .weight_mut_power = self.weight_mut_power,
            .disjoint_coeff = self.disjoint_coeff,
            .excess_coeff = self.excess_coeff,
            .mut_diff_coeff = self.mut_diff_coeff,
            .compat_threshold = self.compat_threshold,
            .age_significance = self.age_significance,
            .survival_thresh = self.survival_thresh,
            .mut_only_prob = self.mut_only_prob,
            .mut_random_trait_prob = self.mut_random_trait_prob,
            .mut_link_trait_prob = self.mut_link_trait_prob,
            .mut_node_trait_prob = self.mut_node_trait_prob,
            .mut_link_weights_prob = self.mut_link_weights_prob,
            .mut_toggle_enable_prob = self.mut_toggle_enable_prob,
            .mut_gene_reenable_prob = self.mut_gene_reenable_prob,
            .mut_add_node_prob = self.mut_add_node_prob,
            .mut_add_link_prob = self.mut_add_link_prob,
            .mut_connect_sensors = self.mut_connect_sensors,
            .interspecies_mate_rate = self.interspecies_mate_rate,
            .mate_multipoint_prob = self.mate_multipoint_prob,
            .mate_multipoint_avg_prob = self.mate_multipoint_avg_prob,
            .mate_singlepoint_prob = self.mate_singlepoint_prob,
            .mate_only_prob = self.mate_only_prob,
            .recur_only_prob = self.recur_only_prob,
            .pop_size = self.pop_size,
            .dropoff_age = self.dropoff_age,
            .new_link_tries = self.new_link_tries,
            .print_every = self.print_every,
            .babies_stolen = self.babies_stolen,
            .num_runs = self.num_runs,
            .num_generations = self.num_generations,
            .epoch_executor_type = self.epoch_executor_type,
            .gen_compat_method = self.gen_compat_method,
            .log_level = self.log_level,
            .node_activators = self.node_activators,
            .node_activators_prob = self.node_activators_prob,
        };
    }

    pub fn write_to_json(self: *Options, path: []const u8) !void {
        var output_file = try get_writable_file(path);
        defer output_file.close();
        try json.toPrettyWriter(null, self.jsonify(), output_file.writer());
    }

    pub fn random_node_activation_type(self: *Options, rand: std.rand.Random) !math.NodeActivationType {
        // quick check for the most cases
        if (self.node_activators.len == 1) {
            return self.node_activators[0];
        }
        // find next random
        var idx = @as(usize, @intCast(m.single_roulette_throw(rand, self.node_activators_prob)));
        if (idx < 0 or idx == self.node_activators.len) {
            std.debug.print("unexpected error when trying to find random node activator, activator index: {d}\n", .{idx});
            return error.NodeActivationTypeInvalid;
        }
        return self.node_activators[idx];
    }

    pub fn init_node_activators(self: *Options, allocator: std.mem.Allocator) !void {
        self.node_activators = try allocator.alloc(math.NodeActivationType, 1);
        self.node_activators[0] = math.NodeActivationType.SigmoidSteepenedActivation;
        self.node_activators_prob = try allocator.alloc(f64, 1);
        self.node_activators_prob[0] = 1.0;
    }

    pub fn deinit(self: *Options) void {
        self.allocator.free(self.node_activators);
        self.allocator.free(self.node_activators_prob);
        self.allocator.destroy(self);
    }

    pub fn read_options(allocator: std.mem.Allocator, path: []const u8) !*Options {
        const buf = try read_file(allocator, path);
        defer allocator.free(buf);

        var self: *Options = try allocator.create(Options);
        self.* = .{ .allocator = allocator };
        var lines = std.mem.split(u8, buf, "\n");
        while (lines.next()) |ln| {
            var split = std.mem.split(u8, ln, " ");
            var name = split.first();
            var value = split.rest();
            if (std.mem.eql(u8, name, "trait_param_mut_prob")) {
                self.trait_param_mut_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "trait_mut_power")) {
                self.trait_mut_power = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "weight_mut_power")) {
                self.weight_mut_power = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "disjoint_coeff")) {
                self.disjoint_coeff = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "excess_coeff")) {
                self.excess_coeff = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_diff_coeff")) {
                self.mut_diff_coeff = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "compat_threshold")) {
                self.compat_threshold = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "age_significance")) {
                self.age_significance = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "survival_thresh")) {
                self.survival_thresh = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_only_prob")) {
                self.mut_only_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_random_trait_prob")) {
                self.mut_random_trait_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_link_trait_prob")) {
                self.mut_link_trait_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_node_trait_prob")) {
                self.mut_node_trait_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_link_weights_prob")) {
                self.mut_link_weights_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_toggle_enable_prob")) {
                self.mut_toggle_enable_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_gene_reenable_prob")) {
                self.mut_gene_reenable_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_add_node_prob")) {
                self.mut_add_node_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_add_link_prob")) {
                self.mut_add_link_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mut_connect_sensors")) {
                self.mut_connect_sensors = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "interspecies_mate_rate")) {
                self.interspecies_mate_rate = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mate_multipoint_prob")) {
                self.mate_multipoint_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mate_multipoint_avg_prob")) {
                self.mate_multipoint_avg_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mate_singlepoint_prob")) {
                self.mate_singlepoint_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "mate_only_prob")) {
                self.mate_only_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "recur_only_prob")) {
                self.recur_only_prob = try std.fmt.parseFloat(f64, value);
            } else if (std.mem.eql(u8, name, "pop_size")) {
                self.pop_size = try std.fmt.parseInt(usize, value, 10);
            } else if (std.mem.eql(u8, name, "dropoff_age")) {
                self.dropoff_age = try std.fmt.parseInt(i64, value, 10);
            } else if (std.mem.eql(u8, name, "new_link_tries")) {
                self.new_link_tries = try std.fmt.parseInt(usize, value, 10);
            } else if (std.mem.eql(u8, name, "print_every")) {
                self.print_every = try std.fmt.parseInt(usize, value, 10);
            } else if (std.mem.eql(u8, name, "babies_stolen")) {
                self.babies_stolen = try std.fmt.parseInt(usize, value, 10);
            } else if (std.mem.eql(u8, name, "num_runs")) {
                self.num_runs = try std.fmt.parseInt(usize, value, 10);
            } else if (std.mem.eql(u8, name, "num_generations")) {
                self.num_generations = try std.fmt.parseInt(usize, value, 10);
            } else if (std.mem.eql(u8, name, "log_level")) {
                const log_levels = [_]std.log.Level{ .info, .debug, .err, .warn };
                for (log_levels) |level| {
                    if (std.mem.eql(u8, @tagName(level), value)) {
                        self.log_level = level;
                        break;
                    }
                }
            } else if (std.mem.eql(u8, name, "epoch_executor_type")) {
                if (std.mem.eql(u8, @tagName(EpochExecutorType.EpochExecutorTypeParallel), value)) {
                    self.epoch_executor_type = .EpochExecutorTypeParallel;
                } else if (std.mem.eql(u8, @tagName(EpochExecutorType.EpochExecutorTypeSequential), value)) {
                    self.epoch_executor_type = .EpochExecutorTypeSequential;
                } else {
                    return error.InvalidEpochExecutorType;
                }
            } else if (std.mem.eql(u8, name, "gen_compat_method")) {
                if (std.mem.eql(u8, @tagName(GenomeCompatibilityMethod.GenomeCompatibilityMethodFast), value)) {
                    self.gen_compat_method = .GenomeCompatibilityMethodFast;
                } else if (std.mem.eql(u8, @tagName(GenomeCompatibilityMethod.GenomeCompatibilityMethodLinear), value)) {
                    self.gen_compat_method = .GenomeCompatibilityMethodLinear;
                } else {
                    return error.InvalidGenCompatMethod;
                }
            }
        }

        // init node activators
        try self.init_node_activators(allocator);
        return self;
    }

    pub fn init_from_json_enc(allocator: std.mem.Allocator, enc: OptionsJSON) !*Options {
        var self = try allocator.create(Options);
        self.* = .{
            .trait_param_mut_prob = enc.trait_param_mut_prob,
            .trait_mut_power = enc.trait_mut_power,
            .weight_mut_power = enc.weight_mut_power,
            .disjoint_coeff = enc.disjoint_coeff,
            .excess_coeff = enc.excess_coeff,
            .mut_diff_coeff = enc.mut_diff_coeff,
            .compat_threshold = enc.compat_threshold,
            .age_significance = enc.age_significance,
            .survival_thresh = enc.survival_thresh,
            .mut_only_prob = enc.mut_only_prob,
            .mut_random_trait_prob = enc.mut_random_trait_prob,
            .mut_link_trait_prob = enc.mut_link_trait_prob,
            .mut_node_trait_prob = enc.mut_node_trait_prob,
            .mut_link_weights_prob = enc.mut_link_weights_prob,
            .mut_toggle_enable_prob = enc.mut_toggle_enable_prob,
            .mut_gene_reenable_prob = enc.mut_gene_reenable_prob,
            .mut_add_node_prob = enc.mut_add_node_prob,
            .mut_add_link_prob = enc.mut_add_link_prob,
            .mut_connect_sensors = enc.mut_connect_sensors,
            .interspecies_mate_rate = enc.interspecies_mate_rate,
            .mate_multipoint_prob = enc.mate_multipoint_prob,
            .mate_multipoint_avg_prob = enc.mate_multipoint_avg_prob,
            .mate_singlepoint_prob = enc.mate_singlepoint_prob,
            .mate_only_prob = enc.mate_only_prob,
            .recur_only_prob = enc.recur_only_prob,
            .pop_size = enc.pop_size,
            .dropoff_age = enc.dropoff_age,
            .new_link_tries = enc.new_link_tries,
            .print_every = enc.print_every,
            .babies_stolen = enc.babies_stolen,
            .num_runs = enc.num_runs,
            .num_generations = enc.num_generations,
            .epoch_executor_type = enc.epoch_executor_type,
            .gen_compat_method = enc.gen_compat_method,
            .log_level = enc.log_level,
            .node_activators = enc.node_activators,
            .node_activators_prob = enc.node_activators_prob,
            .allocator = allocator,
        };

        return self;
    }

    // TODO: maybe??? pub fn validate(self: *Options) !void {}
    pub fn read_from_json(allocator: std.mem.Allocator, path: []const u8) !*Options {
        const buf = try read_file(allocator, path);
        defer allocator.free(buf);
        var enc: OptionsJSON = try json.fromSlice(allocator, OptionsJSON, buf);
        return Options.init_from_json_enc(allocator, enc);
    }
};

// unit test functions & tests for Options

fn check_neat_options(o: *Options) !void {
    try std.testing.expect(o.trait_param_mut_prob == 0.5);
    try std.testing.expect(o.trait_mut_power == 1);
    try std.testing.expect(o.weight_mut_power == 2.5);
    try std.testing.expect(o.disjoint_coeff == 1);
    try std.testing.expect(o.excess_coeff == 1);
    try std.testing.expect(o.mut_diff_coeff == 0.4);
    try std.testing.expect(o.compat_threshold == 3);
    try std.testing.expect(o.age_significance == 1);
    try std.testing.expect(o.survival_thresh == 0.2);
    try std.testing.expect(o.mut_only_prob == 0.25);
    try std.testing.expect(o.mut_random_trait_prob == 0.1);
    try std.testing.expect(o.mut_link_trait_prob == 0.1);
    try std.testing.expect(o.mut_node_trait_prob == 0.1);
    try std.testing.expect(o.mut_link_weights_prob == 0.9);
    try std.testing.expect(o.mut_toggle_enable_prob == 0);
    try std.testing.expect(o.mut_gene_reenable_prob == 0);
    try std.testing.expect(o.mut_add_node_prob == 0.03);
    try std.testing.expect(o.mut_add_link_prob == 0.08);
    try std.testing.expect(o.mut_connect_sensors == 0.5);
    try std.testing.expect(o.interspecies_mate_rate == 0.001);
    try std.testing.expect(o.mate_multipoint_prob == 0.3);
    try std.testing.expect(o.mate_multipoint_avg_prob == 0.3);
    try std.testing.expect(o.mate_singlepoint_prob == 0.3);
    try std.testing.expect(o.mate_only_prob == 0.2);
    try std.testing.expect(o.recur_only_prob == 0);
    try std.testing.expect(o.pop_size == 200);
    try std.testing.expect(o.dropoff_age == 50);
    try std.testing.expect(o.new_link_tries == 50);
    try std.testing.expect(o.print_every == 10);
    try std.testing.expect(o.babies_stolen == 0);
    try std.testing.expect(o.num_runs == 100);
    try std.testing.expect(o.num_generations == 100);
    try std.testing.expect(o.epoch_executor_type == .EpochExecutorTypeSequential);
    try std.testing.expect(o.gen_compat_method == .GenomeCompatibilityMethodFast);
}

const HyperNEATContextJSON = struct {
    neat_ctx: OptionsJSON,
    link_threshold: f64,
    weight_range: f64,
    substrate_activator: math.NodeActivationType,
};

/// The HyperNEAT execution context
pub const HyperNEATContext = struct {
    /// The NEAT context included
    neat_ctx: *Options,

    /// The threshold value to indicate which links should be included
    link_threshold: f64,
    /// The weight range defines the minimum and maximum values for weights on substrate connections,
    ///  they go from -WeightRange to +WeightRange, and can be any integer.
    weight_range: f64,

    /// The substrate activation function
    substrate_activator: math.NodeActivationType,

    allocator: std.mem.Allocator,

    pub fn jsonify(self: *HyperNEATContext) HyperNEATContextJSON {
        return .{
            .neat_ctx = self.neat_ctx.jsonify(),
            .link_threshold = self.link_threshold,
            .weight_range = self.weight_range,
            .substrate_activator = self.substrate_activator,
        };
    }

    pub fn write_to_json(self: *HyperNEATContext, path: []const u8) !void {
        var output_file = try get_writable_file(path);
        defer output_file.close();
        try json.toPrettyWriter(null, self.jsonify(), output_file.writer());
    }

    pub fn init_from_json_enc(allocator: std.mem.Allocator, enc: HyperNEATContextJSON) !*HyperNEATContext {
        var self = try allocator.create(HyperNEATContext);
        self.* = .{
            .neat_ctx = try Options.init_from_json_enc(allocator, enc.neat_ctx),
            .link_threshold = enc.link_threshold,
            .weight_range = enc.weight_range,
            .substrate_activator = enc.substrate_activator,
            .allocator = allocator,
        };
        return self;
    }

    pub fn read_from_json(allocator: std.mem.Allocator, path: []const u8) !*HyperNEATContext {
        const buf = try read_file(allocator, path);
        defer allocator.free(buf);
        var enc: HyperNEATContextJSON = try json.fromSlice(allocator, HyperNEATContextJSON, buf);
        return HyperNEATContext.init_from_json_enc(allocator, enc);
    }

    pub fn deinit(self: *HyperNEATContext) void {
        self.neat_ctx.deinit();
        self.allocator.destroy(self);
    }
};

const ESHyperNEATContextJSON = struct {
    /// The included HyperNEAT context
    hyperneat_ctx: HyperNEATContextJSON,

    /// defines the initial ES-HyperNEAT sample resolution.
    initial_depth: usize,
    /// Maximal ES-HyperNEAT sample resolution if the variance is still higher than the given division threshold
    maximal_depth: usize,

    /// Defines the division threshold. If the variance in a region is greater than this value, after
    /// the initial resolution is reached, ES-HyperNEAT will sample down further (values greater than 1.0 will disable
    /// this feature). Note that sampling at really high resolutions can become computationally expensive.
    division_threshold: f64,
    /// Defines the variance threshold for the initial sampling. The bigger this value the less new
    /// connections will be added directly and the more chances that the new collection will be included in bands
    /// (see banding_threshold).
    variance_threshold: f64,
    /// Defines the threshold that determines when points are regarded to be in a band. If the point
    /// is in the band then no new connection will be added and as result no new hidden node will be introduced.
    /// The bigger this value the less connections/hidden nodes will be added, i.e. wide bands approximation.
    banding_threshold: f64,

    /// Defines how many times ES-HyperNEAT should iteratively discover new hidden nodes.
    es_iterations: usize,
};

pub const ESHyperNEATContext = struct {
    /// The included HyperNEAT context
    hyperneat_ctx: *HyperNEATContext,

    /// defines the initial ES-HyperNEAT sample resolution.
    initial_depth: usize,
    /// Maximal ES-HyperNEAT sample resolution if the variance is still higher than the given division threshold
    maximal_depth: usize,

    /// Defines the division threshold. If the variance in a region is greater than this value, after
    /// the initial resolution is reached, ES-HyperNEAT will sample down further (values greater than 1.0 will disable
    /// this feature). Note that sampling at really high resolutions can become computationally expensive.
    division_threshold: f64,
    /// Defines the variance threshold for the initial sampling. The bigger this value the less new
    /// connections will be added directly and the more chances that the new collection will be included in bands
    /// (see banding_threshold).
    variance_threshold: f64,
    /// Defines the threshold that determines when points are regarded to be in a band. If the point
    /// is in the band then no new connection will be added and as result no new hidden node will be introduced.
    /// The bigger this value the less connections/hidden nodes will be added, i.e. wide bands approximation.
    banding_threshold: f64,

    /// Defines how many times ES-HyperNEAT should iteratively discover new hidden nodes.
    es_iterations: usize,
    allocator: std.mem.Allocator,

    pub fn jsonify(self: *HyperNEATContext) HyperNEATContextJSON {
        return .{
            .hyperneat_ctx = self.hyperneat_ctx.jsonify(),
            .initial_depth = self.initial_depth,
            .maximal_depth = self.maximal_depth,
            .division_threshold = self.division_threshold,
            .variance_threshold = self.variance_threshold,
            .banding_threshold = self.banding_threshold,
            .es_iterations = self.es_iterations,
        };
    }

    pub fn write_to_json(self: *ESHyperNEATContext, path: []const u8) !void {
        var output_file = try get_writable_file(path);
        defer output_file.close();
        try json.toPrettyWriter(null, self.jsonify(), output_file.writer());
    }

    pub fn init_from_json_enc(allocator: std.mem.Allocator, enc: ESHyperNEATContextJSON) !*ESHyperNEATContext {
        var self = try allocator.create(ESHyperNEATContext);
        self.* = .{
            .allocator = allocator,
            .hyperneat_ctx = try HyperNEATContext.init_from_json_enc(allocator, enc.hyperneat_ctx),
            .initial_depth = enc.initial_depth,
            .maximal_depth = enc.maximal_depth,
            .division_threshold = enc.division_threshold,
            .variance_threshold = enc.variance_threshold,
            .banding_threshold = enc.banding_threshold,
            .es_iterations = enc.es_iterations,
        };
        return self;
    }

    pub fn read_from_json(allocator: std.mem.Allocator, path: []const u8) !*ESHyperNEATContext {
        const buf = try read_file(allocator, path);
        defer allocator.free(buf);
        var enc: ESHyperNEATContextJSON = try json.fromSlice(allocator, ESHyperNEATContextJSON, buf);
        return ESHyperNEATContext.init_from_json_enc(allocator, enc);
    }

    pub fn deinit(self: *ESHyperNEATContext) void {
        self.hyperneat_ctx.deinit();
        self.allocator.destroy(self);
    }
};

test "load NEAT Options" {
    var allocator = std.testing.allocator;
    var opts = try Options.read_options(allocator, "examples/xor/data/basic_xor.neat");
    defer opts.deinit();

    try check_neat_options(opts);
}

test "load NEAT Options from JSON" {
    var allocator = std.testing.allocator;
    var opts = try Options.read_from_json(allocator, "data/basic_opts.json");
    defer opts.deinit();

    try check_neat_options(opts);
}

test "load HyperNEATContext from JSON" {
    const allocator = std.testing.allocator;
    var opts = try HyperNEATContext.read_from_json(allocator, "data/basic_hyperneat_opts.json");
    defer opts.deinit();
    try check_neat_options(opts.neat_ctx);
    try std.testing.expect(opts.link_threshold == 0.2);
    try std.testing.expect(opts.weight_range == 3);
    try std.testing.expect(opts.substrate_activator == .SigmoidSteepenedActivation);
}

test "load ESHyperNEATContext from JSON" {
    const allocator = std.testing.allocator;
    var opts = try ESHyperNEATContext.read_from_json(allocator, "data/basic_eshyperneat_opts.json");
    defer opts.deinit();
    try check_neat_options(opts.hyperneat_ctx.neat_ctx);
    try std.testing.expect(opts.hyperneat_ctx.link_threshold == 0.2);
    try std.testing.expect(opts.hyperneat_ctx.weight_range == 3);
    try std.testing.expect(opts.hyperneat_ctx.substrate_activator == .SigmoidSteepenedActivation);
    try std.testing.expect(opts.initial_depth == 3);
    try std.testing.expect(opts.maximal_depth == 5);
    try std.testing.expect(opts.division_threshold == 0.01);
    try std.testing.expect(opts.variance_threshold == 0.03);
    try std.testing.expect(opts.banding_threshold == 0.3);
    try std.testing.expect(opts.es_iterations == 1);
}
