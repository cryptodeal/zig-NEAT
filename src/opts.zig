const std = @import("std");
const math = @import("math/activations.zig");
const m = @import("math/math.zig");
const NeatLogger = @import("log.zig").NeatLogger;
const json = @import("json");

pub const EpochExecutorType = enum {
    EpochExecutorTypeSequential,
    EpochExecutorTypeParallel,
};

pub const GenomeCompatibilityMethod = enum {
    GenomeCompatibilityMethodLinear,
    GenomeCompatibilityMethodFast,
};

pub const logger = &NeatLogger{ .log_level = std.log.Level.info };

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

    pub const @"getty.db" = struct {
        pub const attributes = .{
            .allocator = .{ .skip = true },
        };
    };

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

    pub fn init_node_activators(self: *Options, allocator: std.mem.Allocator, raw: bool) !void {
        if (raw or self.node_activators_with_probs.len == 0) {
            self.node_activators = try allocator.alloc(math.NodeActivationType, 1);
            self.node_activators[0] = math.NodeActivationType.SigmoidSteepenedActivation;
            self.node_activators_prob = try allocator.alloc(f64, 1);
            self.node_activators_prob[0] = 1.0;
            return;
        }
        // create activators
        var act_fns = self.node_activators_with_probs;
        self.node_activators = try allocator.alloc(math.NodeActivationType, act_fns.len);
        self.node_activators_prob = try allocator.alloc(f64, act_fns.len);
        for (act_fns, 0..) |line, i| {
            var field_iterator = std.mem.split(u8, line, " ");
            self.node_activators[i] = math.NodeActivationType.activation_type_by_name(field_iterator.first());
            self.node_activators_prob[i] = try std.fmt.parseFloat(f64, field_iterator.rest());
        }
    }

    pub fn deinit(self: *Options) void {
        self.allocator.free(self.node_activators);
        self.allocator.free(self.node_activators_prob);
        self.allocator.destroy(self);
    }

    pub fn read_options(allocator: std.mem.Allocator, path: []const u8) !*Options {
        const dir_path = std.fs.path.dirname(path);
        const file_name = std.fs.path.basename(path);
        var file_dir: std.fs.Dir = undefined;
        if (dir_path != null) {
            file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
        } else {
            file_dir = std.fs.cwd();
        }
        var file = try file_dir.openFile(file_name, .{});
        const file_size = (try file.stat()).size;
        var buf = try allocator.alloc(u8, file_size);
        defer allocator.free(buf);
        try file.reader().readNoEof(buf);

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
                self.log_level = std.mem.trim(u8, value, &std.ascii.whitespace);
            } else if (std.mem.eql(u8, name, "epoch_executor_type")) {
                if (std.mem.eql(u8, @tagName(EpochExecutorType.EpochExecutorTypeParallel), value)) {
                    self.epoch_executor_type = EpochExecutorType.EpochExecutorTypeParallel;
                } else if (std.mem.eql(u8, @tagName(EpochExecutorType.EpochExecutorTypeSequential), value)) {
                    self.epoch_executor_type = EpochExecutorType.EpochExecutorTypeSequential;
                } else {
                    return error.InvalidEpochExecutorType;
                }
            } else if (std.mem.eql(u8, name, "gen_compat_method")) {
                if (std.mem.eql(u8, @tagName(GenomeCompatibilityMethod.GenomeCompatibilityMethodFast), value)) {
                    self.gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodFast;
                } else if (std.mem.eql(u8, @tagName(GenomeCompatibilityMethod.GenomeCompatibilityMethodLinear), value)) {
                    self.gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodLinear;
                } else {
                    return error.InvalidGenCompatMethod;
                }
            }
        }

        // read node activators
        try self.init_node_activators(allocator, true);
        return self;
    }

    // TODO: maybe??? pub fn validate(self: *Options) !void {}
    pub fn load_json_opts(allocator: std.mem.Allocator, path: []const u8) !*Options {
        const dir_path = std.fs.path.dirname(path);
        const file_name = std.fs.path.basename(path);
        var file_dir: std.fs.Dir = undefined;
        if (dir_path != null) {
            file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
        } else {
            file_dir = std.fs.cwd();
        }
        var file = try file_dir.openFile(file_name, .{});
        const file_size = (try file.stat()).size;
        var buf = try allocator.alloc(u8, file_size);
        defer allocator.free(buf);
        try file.reader().readNoEof(buf);

        var self: Options = try json.fromSlice(allocator, Options, buf);

        self.allocator = allocator;

        // read node activators
        try self.init_node_activators(allocator, false);

        return &self;
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
    try std.testing.expect(o.epoch_executor_type == EpochExecutorType.EpochExecutorTypeSequential);
    try std.testing.expect(o.gen_compat_method == GenomeCompatibilityMethod.GenomeCompatibilityMethodFast);
}

test "load NEAT Options" {
    var allocator = std.testing.allocator;
    var opts = try Options.read_options(allocator, "examples/xor/data/basic_xor.neat");
    defer opts.deinit();

    try check_neat_options(opts);
}
