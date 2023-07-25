const std = @import("std");
const zig_neat = @import("zigNEAT");
const common = @import("common.zig");
const env = @import("environment.zig");
const stores = @import("maze_data_store.zig");
const ns = @import("maze_ns.zig");

const archive_thresh = ns.archive_thresh;
const novelty_metric = ns.novelty_metric;
const NeatLogger = zig_neat.NeatLogger;
const hist_diff = common.hist_diff;
const maze_simulation_evaluate = common.maze_simulation_evaluate;
const adjust_species_number = common.adjust_species_number;
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
const create_out_dir_for_trial = zig_neat.experiment.create_out_dir_for_trial;

const Options = zig_neat.Options;

var logger = NeatLogger{ .log_level = std.log.Level.info };

pub const MazeObjectiveEvaluator = struct {
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

    pub fn trial_run_started(self: *MazeObjectiveEvaluator, trial: *Trial) void {
        var opts = NoveltyArchiveOptions.init(self.allocator) catch |err| {
            logger.err("Failed to initialize novelty archive options: {any}", .{err}, @src());
            return;
        };
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

    pub fn trial_run_finished(self: *MazeObjectiveEvaluator, trial: *Trial) void {
        _ = trial;
        // the last epoch executed
        self.store_recorded(self.allocator) catch |err| {
            logger.err("Failed to store recorded data: {any}", .{err}, @src());
            return;
        };
        self.trial_sim.deinit();
    }

    pub fn epoch_evaluated(self: *MazeObjectiveEvaluator, trial: *Trial, epoch: *Generation) void {
        _ = self;
        _ = trial;
        _ = epoch;
    }

    fn store_recorded(self: *MazeObjectiveEvaluator, allocator: std.mem.Allocator) !void {
        // store recorded agents' performance
        var buf = std.ArrayList(u8).init(allocator);
        defer buf.deinit();
        try create_out_dir_for_trial(buf.writer(), self.output_path, self.trial_sim.trial_id);
        try buf.appendSlice("/record.dat");
        var dir_path = std.fs.path.dirname(buf.items);
        var file_name = std.fs.path.basename(buf.items);
        var file_dir: std.fs.Dir = undefined;
        if (dir_path != null) {
            file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
        } else {
            file_dir = std.fs.cwd();
        }
        var output_file = try file_dir.createFile(file_name, .{});
        defer output_file.close();
        try self.trial_sim.records.write(output_file.writer());
        buf.clearAndFree(); // reset path

        // print novelty points with maximal fitness
        try create_out_dir_for_trial(buf.writer(), self.output_path, self.trial_sim.trial_id);
        try buf.appendSlice("/fittest_archive_points.json");
        try self.trial_sim.archive.dump_fittest(buf.items);
    }

    pub fn generation_evaluate(self: *MazeObjectiveEvaluator, opts: *Options, pop: *Population, epoch: *Generation) !void {
        // Evaluate each organism on a test
        for (pop.organisms.items) |org| {
            var res = try self.org_eval(self.allocator, org, epoch);
            if (res and (epoch.champion == null or org.fitness > epoch.champion.?.fitness)) {
                epoch.solved = true;
                epoch.winner_nodes = org.genotype.nodes.len;
                epoch.winner_genes = @as(usize, @intCast(org.genotype.extrons()));
                epoch.winner_evals = self.trial_sim.individuals_counter;
                epoch.champion = org;
            }
        }

        // Fill statistics about current epoch
        try epoch.fill_population_statistics(pop);

        // TODO: Only print to file every print_every generation

        if (epoch.solved) {
            var org: *Organism = epoch.champion.?;
            std.debug.print("Winner organism fitness: {d}\n", .{org.fitness});
            var depth = try org.phenotype.?.max_activation_depth_capped(0);
            std.debug.print("Activation depth of the winner: {d}\n", .{depth});
        } else if (epoch.id < opts.num_generations - 1) {
            var species_count = pop.species.items.len;

            // adjust species count by keeping it constant
            try adjust_species_number(species_count, epoch.id, self.compat_adjust_freq, self.num_species_target, opts);
            logger.info("{d} species -> {d} organisms [compatibility threshold: {d:.1}, target: {d}]", .{ species_count, pop.organisms.items.len, opts.compat_threshold, self.num_species_target }, @src());
        }
    }

    /// Evaluates individual organism against maze environment and returns true if organism was able to solve maze by navigating to exit
    fn org_eval(self: *MazeObjectiveEvaluator, allocator: std.mem.Allocator, org: *Organism, epoch: *Generation) !bool {
        // create record to store simulation results for organism
        var record = try AgentRecord.init(allocator);
        record.generation = epoch.id;
        record.agent_id = self.trial_sim.individuals_counter;
        record.species_id = @as(usize, @intCast(org.species.id));
        record.species_age = @as(usize, @intCast(org.species.age));

        // evaluate individual organism and get novelty point holding simulation results
        var eval_res = maze_simulation_evaluate(allocator, self.maze_env, org, record, null) catch |err| {
            if (err == error.OutputIsNaN) {
                return false;
            }
            return err;
        };
        var n_item = eval_res.item;
        var solved = eval_res.exit_found;
        try self.trial_sim.archive.encountered_items.append(n_item); // track all novelty items

        n_item.individual_id = @as(usize, @intCast(org.genotype.id));
        // assign organism fitness based on simulation results - the normalized distance between agent and maze exit
        org.fitness = n_item.fitness;
        org.is_winner = solved;
        org.error_value = 1 - n_item.fitness;

        if (solved) {
            // run simulation to store solver path points
            var path_points = try std.ArrayList(*Point).initCapacity(allocator, self.maze_env.time_steps);
            var tmp_res = try maze_simulation_evaluate(allocator, self.maze_env, org, null, &path_points);
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

        // update the fittest organisms list - needed for debugging output
        org.data = n_item;
        try self.trial_sim.archive.update_fittest_with_organism(allocator, org);

        return solved;
    }
};
