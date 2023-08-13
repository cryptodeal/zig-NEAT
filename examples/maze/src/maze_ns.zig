const std = @import("std");
const zig_neat = @import("zigNEAT");
const common = @import("common.zig");
const env = @import("environment.zig");
const stores = @import("maze_data_store.zig");

const NeatLogger = zig_neat.NeatLogger;
const histDiff = common.histDiff;
const mazeSimulationEvaluate = common.mazeSimulationEvaluate;
const adjustSpeciesNumber = common.adjustSpeciesNumber;
const Environment = env.Environment;
const Point = env.Point;
const MazeSimResults = common.MazeSimResults;
const AgentRecord = stores.AgentRecord;
const RecordStore = stores.RecordStore;
const NoveltyItem = zig_neat.ns.NoveltyItem;
const NoveltyArchive = zig_neat.ns.NoveltyArchive;
const NoveltyArchiveOptions = zig_neat.ns.NoveltyArchiveOptions;
const Trial = zig_neat.experiment.Trial;
const Generation = zig_neat.experiment.Generation;
const Organism = zig_neat.genetics.Organism;
const Population = zig_neat.genetics.Population;
const createOutDirForTrial = zig_neat.experiment.createOutDirForTrial;

const Options = zig_neat.Options;

var logger = NeatLogger{ .log_level = std.log.Level.info };

/// the initial novelty threshold for NoveltyArchive
pub const archive_thresh: f64 = 6;

/// the novelty metric function for maze simulation
pub fn novelty_metric(x: *NoveltyItem, y: *NoveltyItem) f64 {
    return histDiff(f64, x.data.items, y.data.items);
}

pub const MazeNsGenerationEvaluator = struct {
    /// the output path to store execution results
    output_path: []const u8,
    /// maze seed environment
    maze_env: *Environment,
    /// the target number of species to be maintained
    num_species_target: usize,
    /// the species compatibility threshold adjustment frequency
    compat_adjust_freq: usize,

    trial_sim: *MazeSimResults = undefined,

    allocator: std.mem.Allocator,

    /// Evaluates individual organism against maze environment and returns true if organism was able to solve maze by navigating to exit
    fn orgEval(self: *MazeNsGenerationEvaluator, allocator: std.mem.Allocator, org: *Organism, pop: *Population, epoch: *Generation) !bool {
        // create record to store simulation results for organism
        var record = try AgentRecord.init(allocator);
        errdefer record.deinit();
        record.generation = epoch.id;
        record.agent_id = self.trial_sim.individuals_counter;
        record.species_id = @as(usize, @intCast(org.species.id));
        record.species_age = @as(usize, @intCast(org.species.age));

        // evaluate individual organism and get novelty point
        var eval_res = mazeSimulationEvaluate(allocator, self.maze_env, org, record, null) catch |err| {
            if (err == error.OutputIsNaN) {
                return false;
            }
            return err;
        };
        var n_item = eval_res.item;
        var solved = eval_res.exit_found;
        try self.trial_sim.archive.encountered_items.append(n_item);

        n_item.individual_id = @as(u64, @intCast(org.genotype.id));
        org.data = n_item; // store novelty item within organism data
        org.is_winner = solved; // store if maze was solved
        org.error_value = 1 - n_item.fitness; // error value consider how far  we are from exit normalized to (0;1] range

        // calculate novelty of new individual within archive of known novel items
        if (!solved) {
            try self.trial_sim.archive.evaluateIndividualNovelty(allocator, org, pop, false);
            record.novelty = n_item.novelty; // add novelty value to the record
        } else {
            // solution found - set to maximal possible value
            record.novelty = std.math.inf(f64);

            // run simulation to store solver path
            var path_points = try std.ArrayList(*Point).initCapacity(allocator, self.maze_env.time_steps);
            var tmp_res = try mazeSimulationEvaluate(allocator, self.maze_env, org, null, &path_points);
            tmp_res.item.deinit();
            for (self.trial_sim.records.solver_path_points.items) |p| {
                p.deinit();
            }
            self.trial_sim.records.solver_path_points.deinit();
            self.trial_sim.records.solver_path_points = path_points;
        }

        // add record
        try self.trial_sim.records.records.append(record);

        // increment tested unique individuals counter
        self.trial_sim.individuals_counter += 1;

        // update fittest organisms list
        try self.trial_sim.archive.updateFittestWithOrganism(allocator, org);

        return solved;
    }

    pub fn generationEvaluate(self: *MazeNsGenerationEvaluator, opts: *Options, pop: *Population, epoch: *Generation) !void {
        // evaluate each organism on a test
        for (pop.organisms.items) |org| {
            var res = try self.orgEval(self.allocator, org, pop, epoch);
            // store fitness based on objective proximity for statistical purposes
            if (org.data == null) {
                logger.err("Novelty point not found at organism: {any}", .{org}, @src());
                org.fitness = 0.0;
            } else {
                var data: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
                org.fitness = data.fitness;
            }

            if (res and (epoch.champion == null or org.fitness > epoch.champion.?.fitness)) {
                epoch.solved = true;
                epoch.winner_nodes = org.genotype.nodes.len;
                epoch.winner_genes = @as(usize, @intCast(org.genotype.extrons()));
                epoch.winner_evals = self.trial_sim.individuals_counter;
                epoch.champion = org;
            }
        }

        // Fill statistics about current epoch
        try epoch.fillPopulationStatistics(pop);

        // TODO: Only print to file every print_every generation

        if (epoch.solved) {
            // print winner organism
            var org: *Organism = epoch.champion.?;
            std.debug.print("Winner organism fitness: {d}\n", .{org.fitness});

            var depth = try org.phenotype.?.maxActivationDepthCapped(0);
            std.debug.print("Activation depth of the winner: {d}\n", .{depth});
        } else if (epoch.id < opts.num_generations - 1) {
            // adjust archive settings
            self.trial_sim.archive.endOfGeneration();
            // refresh generation's novelty scores
            try self.trial_sim.archive.evaluatePopulationNovelty(self.allocator, pop, true);

            var species_count = pop.species.items.len;

            // adjust species count by keeping it constant
            try adjustSpeciesNumber(species_count, epoch.id, self.compat_adjust_freq, self.num_species_target, opts);
            logger.info("{d} species -> {d} organisms [compatibility threshold: {d:.1}, target: {d}]\n", .{ species_count, pop.organisms.items.len, opts.compat_threshold, self.num_species_target }, @src());
        }
    }

    pub fn trialRunStarted(self: *MazeNsGenerationEvaluator, trial: *Trial) void {
        var opts = NoveltyArchiveOptions.init(self.allocator) catch |err| {
            logger.err("Failed to initialize novelty archive options: {any}", .{err}, @src());
            return;
        };
        opts.knn_novelty_score = 10;
        var record_store = RecordStore.init(self.allocator) catch |err| {
            logger.err("Failed to initialize novelty archive options: {any}", .{err}, @src());
            return;
        };
        var archive = NoveltyArchive.init(self.allocator, archive_thresh, novelty_metric, opts) catch |err| {
            logger.err("Failed to initialize novelty archive options: {any}", .{err}, @src());
            return;
        };
        self.trial_sim = MazeSimResults.init(self.allocator, record_store, archive, @as(usize, @intCast(trial.id))) catch |err| {
            logger.err("Failed to initialize novelty archive options: {any}", .{err}, @src());
            return;
        };
    }

    pub fn trialRunFinished(self: *MazeNsGenerationEvaluator, trial: *Trial) void {
        _ = trial;
        // the last epoch executed
        self.storeRecorded(self.allocator) catch |err| {
            logger.err("Failed to store recorded data: {any}", .{err}, @src());
            return;
        };
        self.trial_sim.deinit();
    }

    pub fn epochEvaluated(self: *MazeNsGenerationEvaluator, trial: *Trial, epoch: *Generation) void {
        _ = self;
        _ = trial;
        _ = epoch;
    }

    fn storeRecorded(self: *MazeNsGenerationEvaluator, allocator: std.mem.Allocator) !void {
        // store recorded agents' performance
        var buf = std.ArrayList(u8).init(allocator);
        defer buf.deinit();
        try createOutDirForTrial(buf.writer(), self.output_path, self.trial_sim.trial_id);
        try buf.appendSlice("/record.dat");
        var dir_path = std.fs.path.dirname(buf.items);
        var file_name = std.fs.path.basename(buf.items);
        var file_dir: std.fs.Dir = undefined;
        defer file_dir.close();
        if (dir_path != null) {
            file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
        } else {
            file_dir = std.fs.cwd();
        }
        var output_file = try file_dir.createFile(file_name, .{});
        defer output_file.close();
        try self.trial_sim.records.write(output_file.writer());
        buf.clearAndFree(); // reset path

        // print collected novelty points from archive
        try createOutDirForTrial(buf.writer(), self.output_path, self.trial_sim.trial_id);
        try buf.appendSlice("/novelty_archive_points.json");
        try self.trial_sim.archive.dumpNoveltyPoints(buf.items);
        buf.clearAndFree(); // reset path

        // print novelty points with maximal fitness
        try createOutDirForTrial(buf.writer(), self.output_path, self.trial_sim.trial_id);
        try buf.appendSlice("/fittest_novelty_archive_points.json");
        try self.trial_sim.archive.dumpFittest(buf.items);
    }
};
