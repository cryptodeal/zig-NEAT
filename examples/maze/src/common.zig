const std = @import("std");
const store = @import("maze_data_store.zig");
const maze_env = @import("environment.zig");
const zig_neat = @import("zigNEAT");

const NeatLogger = zig_neat.NeatLogger;

const RecordStore = store.RecordStore;
const AgentRecord = store.AgentRecord;
const Environment = maze_env.Environment;
const Point = maze_env.Point;
const Options = zig_neat.Options;
const Organism = zig_neat.genetics.Organism;
const NoveltyArchive = zig_neat.ns.NoveltyArchive;
const NoveltyItem = zig_neat.ns.NoveltyItem;

pub const compatibility_threshold_step: f64 = 0.1;
pub const compatibility_threshold_min_value: f64 = 0.3;

var logger = NeatLogger{ .log_level = std.log.Level.info };

/// The structure to hold maze simulator evaluation results
pub const MazeSimResults = struct {
    /// The record store for evaluated agents
    records: *RecordStore,

    /// The novelty archive
    archive: *NoveltyArchive,

    /// the current trial
    trial_id: usize,
    /// The evaluated individuals counter within current trial
    individuals_counter: usize = 0,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, records: *RecordStore, archive: *NoveltyArchive, trial_id: usize) !*MazeSimResults {
        var self = try allocator.create(MazeSimResults);
        self.* = .{
            .records = records,
            .archive = archive,
            .trial_id = trial_id,
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *MazeSimResults) void {
        self.records.deinit();
        self.archive.deinit();
        self.allocator.destroy(self);
    }
};

/// calculates item-wise difference between two vectors (of floats)
pub fn histDiff(comptime T: type, left: []T, right: []T) T {
    var diff_accum: T = 0;
    for (left, 0..) |_, i| {
        diff_accum += @fabs(left[i] - right[i]);
    }
    return diff_accum / @as(T, @floatFromInt(left.len));
}

pub const MazeEvalResults = struct {
    item: *NoveltyItem,
    exit_found: bool,
};

pub fn mazeSimulationEvaluate(allocator: std.mem.Allocator, env: *Environment, org: *Organism, record: ?*AgentRecord, path_points: ?*std.ArrayList(*Point)) !MazeEvalResults {
    var n_item = try NoveltyItem.init(allocator);
    errdefer n_item.deinit();

    // get Organism phenotype's network depth
    var net_depth = try org.phenotype.?.maxActivationDepthCapped(1); // The max depth of the network to be activated
    logger.debug("Network depth: {d} for organism: {d}\n", .{ net_depth, org.genotype.id }, @src());
    if (net_depth == 0) {
        logger.debug("ALERT: Network depth is ZERO for Genome: {s}", .{org.genotype}, @src());
    }

    // initialize maze simulation's environment specific to the provided organism - this will be a copy
    // of primordial environment provided
    var org_env = try mazeSimulationInit(allocator, env, org, net_depth);
    defer org_env.deinit();

    // do a specified amount of time steps emulations or while exit not found
    var steps: usize = 0;
    var i: usize = 0;
    while (i < org_env.time_steps and !org_env.exit_found) : (i += 1) {
        try mazeSimulationStep(allocator, org_env, org, net_depth);

        // store agent path points at given sample size
        if (try std.math.mod(usize, org_env.time_steps - i, org_env.sample_size) == 0) {
            try n_item.data.append(org_env.hero.location.x);
            try n_item.data.append(org_env.hero.location.y);
        }

        if (path_points != null) {
            path_points.?.appendAssumeCapacity(try org_env.hero.location.clone(allocator));
        }
        steps += 1;
    }

    if (org_env.exit_found) {
        logger.info("Maze solved in: {d} steps\n", .{steps}, @src());
    }

    // calculate fitness of an organism as closeness to target
    var fitness = org_env.agentDistanceToExit();

    // normalize fitness value in range (0;1] and store it
    fitness = (env.initial_distance - fitness) / env.initial_distance;
    if (fitness <= 0) {
        fitness = 0.01;
    }
    n_item.fitness = fitness;
    try n_item.data.append(org_env.hero.location.x);
    try n_item.data.append(org_env.hero.location.y);

    // store final agent coordinates as organism's novelty characteristics
    if (record != null) {
        record.?.fitness = fitness;
        record.?.x = org_env.hero.location.x;
        record.?.y = org_env.hero.location.y;
        record.?.got_exit = org_env.exit_found;
    }

    return MazeEvalResults{ .item = n_item, .exit_found = org_env.exit_found };
}

pub fn mazeSimulationInit(allocator: std.mem.Allocator, env: *Environment, org: *Organism, net_depth: i64) !*Environment {
    var env_copy = try env.clone(allocator);
    errdefer env_copy.deinit();

    // flush the neural net
    _ = try org.phenotype.?.flush();

    // update the maze
    try env_copy.update();

    // create neural net inputs from environment
    var inputs = try env_copy.getInputs(allocator);
    defer allocator.free(inputs);
    org.phenotype.?.loadSensors(inputs);

    // propagate input through the phenotype net

    // Use depth to ensure full relaxation
    _ = org.phenotype.?.forwardSteps(net_depth) catch |err| {
        if (err != error.ErrNetExceededMaxActivationAttempts) {
            logger.err("Failed to activate network at call to `forwardSteps`", .{}, @src());
            return err;
        }
    };

    return env_copy;
}

/// executes a time step of the maze simulation evaluation within given Environment for provided Organism
pub fn mazeSimulationStep(allocator: std.mem.Allocator, env: *Environment, org: *Organism, net_depth: i64) !void {
    // get simulation parameters as inputs to organism's network
    var inputs = try env.getInputs(allocator);
    defer allocator.free(inputs);

    org.phenotype.?.loadSensors(inputs);

    _ = org.phenotype.?.forwardSteps(net_depth) catch |err| {
        if (err != error.ErrNetExceededMaxActivationAttempts) {
            logger.err("Failed to activate network at call to `forwardSteps`", .{}, @src());
            return err;
        }
    };

    // use the net's outputs to change heading and velocity of maze agent
    env.applyOutputs(org.phenotype.?.outputs[0].activation, org.phenotype.?.outputs[1].activation) catch |err| {
        logger.err("Failed to apply outputs", .{}, @src());
        return err;
    };

    // update the environment
    env.update() catch |err| {
        logger.err("Failed to update environment", .{}, @src());
        return err;
    };
}

/// used to adjust species count by keeping it constant
pub fn adjustSpeciesNumber(species_count: usize, epoch_id: usize, adjust_frequency: usize, number_species_target: usize, options: *Options) !void {
    if (try std.math.mod(usize, epoch_id, adjust_frequency) == 0) {
        if (species_count < number_species_target) {
            options.compat_threshold -= compatibility_threshold_step;
        } else if (species_count > number_species_target) {
            options.compat_threshold += compatibility_threshold_step;
        }

        // to avoid dropping too low
        if (options.compat_threshold < compatibility_threshold_min_value) {
            options.compat_threshold = compatibility_threshold_min_value;
        }
    }
}

test "adjustSpeciesNumber" {
    var initial_threshold: f64 = 0.5;
    var options = Options{ .compat_threshold = initial_threshold };

    // check no changes
    var epoch_id: usize = 1;
    var adjust_frequency: usize = 5;
    var species_count: usize = 10;
    var number_species_target: usize = 20;
    try adjustSpeciesNumber(species_count, epoch_id, adjust_frequency, number_species_target, &options);
    try std.testing.expect(options.compat_threshold == initial_threshold);

    // check species_count < number_species_target
    epoch_id = adjust_frequency;
    try adjustSpeciesNumber(species_count, epoch_id, adjust_frequency, number_species_target, &options);
    try std.testing.expect(options.compat_threshold == initial_threshold - compatibility_threshold_step);

    // check species_count > number_species_target
    options.compat_threshold = initial_threshold;
    species_count = number_species_target + 1;
    try adjustSpeciesNumber(species_count, epoch_id, adjust_frequency, number_species_target, &options);
    try std.testing.expect(options.compat_threshold == initial_threshold + compatibility_threshold_step);

    // check speciesCount == numberSpeciesTarget
    options.compat_threshold = initial_threshold;
    species_count = number_species_target;
    try adjustSpeciesNumber(species_count, epoch_id, adjust_frequency, number_species_target, &options);
    try std.testing.expect(options.compat_threshold == initial_threshold);

    // check avoiding of dropping too low
    options.compat_threshold = compatibility_threshold_min_value;
    species_count = number_species_target - 1;
    try adjustSpeciesNumber(species_count, epoch_id, adjust_frequency, number_species_target, &options);
    try std.testing.expect(options.compat_threshold == compatibility_threshold_min_value);
}

test "histDiff" {
    var left = [_]f64{ 1, 2, 3, 4 };
    var right = [_]f64{ 4, 3, 2, 1 };
    var diff = histDiff(f64, &left, &right); // (3 + 1 + 1 + 3) / 4 = 2
    try std.testing.expect(diff == 2);
}

test {
    std.testing.refAllDecls(@This());
}
