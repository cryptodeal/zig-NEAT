const std = @import("std");
const novelty_item = @import("novelty_item.zig");
const common = @import("common.zig");
const opt = @import("../opts.zig");
const json = @import("json");

const NoveltyMetric = common.NoveltyMetric;
const NoveltyArchiveOptions = common.NoveltyArchiveOptions;
const Organism = @import("../genetics/organism.zig").Organism;
const Population = @import("../genetics/population.zig").Population;
const Genome = @import("../genetics/genome.zig").Genome;
const NoveltyItem = novelty_item.NoveltyItem;
const ItemsDistance = novelty_item.ItemsDistance;
const Options = opt.Options;
const novelty_item_comparison = novelty_item.novelty_item_comparison;
const items_distance_comparison = novelty_item.items_distance_comparison;
const logger = @constCast(opt.logger);

/// The novelty archive contains all the novel items we have encountered thus far.
/// Using a novelty metric we can determine how novel a new item is compared to everything
/// currently in the novelty set
pub const NoveltyArchive = struct {
    /// all the novel items we have found so far
    novel_items: std.ArrayList(*NoveltyItem),
    /// all novel items from the fittest organisms found so far
    fittest_items: std.ArrayList(*NoveltyItem),

    /// the current generation
    generation: usize = 0,

    /// the measure of novelty
    novelty_metric: *const NoveltyMetric,

    /// the novel items added during current generation
    items_added_in_generation: usize = 0,
    /// the current generation index
    generation_index: usize,

    /// the minimum threshold for a "novel item"
    novelty_threshold: f64,
    /// the minimal possible value of novelty threshold
    novelty_floor: f64 = 0.25,

    /// the counter to keep track of how many gens since we've added to the archive
    timeout: usize = 0,

    options: *NoveltyArchiveOptions,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, threshold: f64, metric: *const NoveltyMetric, options: *NoveltyArchiveOptions) !*NoveltyArchive {
        var self = try allocator.create(NoveltyArchive);
        self.* = .{
            .allocator = allocator,
            .novel_items = std.ArrayList(*NoveltyItem).init(allocator),
            .fittest_items = std.ArrayList(*NoveltyItem).init(allocator),
            .novelty_metric = metric,
            .novelty_threshold = threshold,
            .generation_index = options.archive_seed_amount,
            .options = options,
        };
        return self;
    }

    pub fn deinit(self: *NoveltyArchive) void {
        self.novel_items.deinit();
        self.fittest_items.deinit();
        self.options.deinit();
        self.allocator.destroy(self);
    }

    /// Evaluates the novelty of a single individual organism within population and update
    /// its fitness (only_fitness = true) or store individual's novelty item into archive
    pub fn evaluate_individual_novelty(self: *NoveltyArchive, allocator: std.mem.Allocator, org: *Organism, pop: *Population, only_fitness: bool) !void {
        if (org.data == null) {
            logger.info("WARNING! Found Organism without novelty point associated: {any}\nNovelty evaluation will be skipped for it. Probably winner found!", .{org}, @src());
            return;
        }

        var item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
        var result: f64 = undefined;
        if (only_fitness) {
            // assign organism fitness according to average novelty within archive and population
            result = try self.novelty_avg_knn(allocator, item, -1, pop);
            org.fitness = result;
        } else {
            // consider adding a point to archive based on dist to nearest neighbor
            result = try self.novelty_avg_knn(allocator, item, 1, null);
            if (result > self.novelty_threshold or self.novel_items.items.len < self.options.archive_seed_amount) {
                try self.add_novelty_item(item);
                item.age += 1;
            }
        }

        // store found values to the item
        item.novelty = result;
        item.generation = self.generation;
        org.data = item;
    }

    /// Evaluates the novelty of the whole population and update organisms fitness (only_fitness = true)
    /// or store each population individual's novelty items into archive
    pub fn evaluate_population_novelty(self: *NoveltyArchive, allocator: std.mem.Allocator, pop: *Population, only_fitness: bool) !void {
        for (pop.organisms.items) |org| {
            try self.evaluate_individual_novelty(allocator, org, pop, only_fitness);
        }
    }

    pub fn update_fittest_with_organism(self: *NoveltyArchive, allocator: std.mem.Allocator, org: *Organism) !void {
        if (org.data == null) {
            logger.err("organism with no data provided", .{}, @src());
            return error.OrganismHasNoData;
        }

        if (self.fittest_items.items.len < self.options.fittest_allowed_size) {
            // store organism's novelty item into fittest
            var item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
            try self.fittest_items.append(item);

            // sort to have the most fit first
            std.mem.sort(*NoveltyItem, self.fittest_items.items, {}, novelty_item_comparison);
            std.mem.reverse(*NoveltyItem, self.fittest_items.items);
        } else {
            var last_item: *NoveltyItem = self.fittest_items.items[self.fittest_items.items.len - 1];
            var org_item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
            if (org_item.fitness > last_item.fitness) {
                // store organism's novelty item into fittest
                try self.fittest_items.append(org_item);

                // sort to have the most fit first
                std.mem.sort(*NoveltyItem, self.fittest_items.items, {}, novelty_item_comparison);
                std.mem.reverse(*NoveltyItem, self.fittest_items.items);

                // remove less fit item
                var items = try std.ArrayList(*NoveltyItem).initCapacity(allocator, self.options.fittest_allowed_size);
                items.expandToCapacity();
                @memcpy(items.items, self.fittest_items.items[0..self.options.fittest_allowed_size]);
                self.fittest_items.deinit();
                self.fittest_items = items;
            }
        }
    }

    /// the steady-state end of generation call
    pub fn end_of_generation(self: *NoveltyArchive) void {
        self.generation += 1;
        self.adjust_archive_settings();
    }

    /// adds novelty item to archive
    pub fn add_novelty_item(self: *NoveltyArchive, i: *NoveltyItem) !void {
        i.added = true;
        i.generation = self.generation;
        try self.novel_items.append(i);
        self.items_added_in_generation += 1;
    }

    /// Used to adjust dynamic novelty threshold depending on how many have been added to archive recently
    pub fn adjust_archive_settings(self: *NoveltyArchive) void {
        if (self.items_added_in_generation == 0) {
            self.timeout += 1;
        } else {
            self.timeout = 0;
        }

        // if no individuals have been added for 10 generations lower the threshold
        if (self.timeout == 10) {
            self.novelty_threshold *= 0.95;
            if (self.novelty_threshold < self.novelty_floor) {
                self.novelty_threshold = self.novelty_floor;
            }
            self.timeout = 0;
        }

        // if more than four individuals added this generation raise threshold
        if (self.items_added_in_generation >= 4) {
            self.novelty_threshold *= 1.2;
        }

        self.items_added_in_generation = 0;
        self.generation_index = self.novel_items.items.len;
    }

    /// Allows the K nearest neighbor novelty score calculation for given item within provided population
    pub fn novelty_avg_knn(self: *NoveltyArchive, allocator: std.mem.Allocator, item: *NoveltyItem, neighbors: i64, pop: ?*Population) !f64 {
        var novelties: []*ItemsDistance = undefined;
        if (pop != null) {
            novelties = try self.map_novelty_in_population(allocator, item, pop.?);
        } else {
            novelties = try self.map_novelty(allocator, item);
        }
        defer if (novelties.len > 0) {
            defer allocator.free(novelties);
            for (novelties) |nov| {
                nov.deinit();
            }
        };

        // sort by distance - minimal first
        std.mem.sort(*ItemsDistance, novelties, {}, items_distance_comparison);
        var length = novelties.len;
        var used_neighbors: usize = undefined;
        if (neighbors < 0) {
            used_neighbors = self.options.knn_novelty_score;
        } else {
            used_neighbors = @as(usize, @intCast(neighbors));
        }

        var density: f64 = 0;
        if (length >= self.options.archive_seed_amount) {
            var sum: f64 = 0;
            var count: f64 = 0;
            var i: usize = 0;
            while (count < @as(f64, @floatFromInt(used_neighbors)) and i < length) : (i += 1) {
                sum += novelties[i].distance;
                count += 1;
            }

            // find average
            if (count > 0) {
                density = sum / count;
            }
        }
        return density;
    }

    /// maps the novelty metric across the archive against provided item
    pub fn map_novelty(self: *NoveltyArchive, allocator: std.mem.Allocator, item: *NoveltyItem) ![]*ItemsDistance {
        var distances = try allocator.alloc(*ItemsDistance, self.novel_items.items.len);
        for (self.novel_items.items, 0..) |nov, i| {
            distances[i] = try ItemsDistance.init(allocator, nov, item, self.novelty_metric(nov, item));
        }
        return distances;
    }

    /// Maps the novelty metric across the archive and the current population
    pub fn map_novelty_in_population(self: *NoveltyArchive, allocator: std.mem.Allocator, item: *NoveltyItem, pop: *Population) ![]*ItemsDistance {
        var distances = try std.ArrayList(*ItemsDistance).initCapacity(allocator, self.novel_items.items.len);

        for (self.novel_items.items) |nov| {
            var dist = try ItemsDistance.init(allocator, nov, item, self.novelty_metric(nov, item));
            distances.appendAssumeCapacity(dist);
        }

        for (pop.organisms.items) |org| {
            if (org.data == null) continue;
            var org_item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
            var dist = try ItemsDistance.init(allocator, org_item, item, self.novelty_metric(org_item, item));
            try distances.append(dist);
        }

        return distances.toOwnedSlice();
    }

    /// dumps collected novelty points to file as JSON
    pub fn dump_novelty_points(self: *NoveltyArchive, path: []const u8) !void {
        if (self.novel_items.items.len == 0) return error.NoNovelItems;

        const dir_path = std.fs.path.dirname(path);
        const file_name = std.fs.path.basename(path);
        var file_dir: std.fs.Dir = undefined;
        if (dir_path != null) {
            file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
        } else {
            file_dir = std.fs.cwd();
        }
        var output_file = try file_dir.createFile(file_name, .{});
        defer output_file.close();
        try self.dump_novelty_items(self.novel_items.items, output_file.writer());
    }

    /// dumps collected novelty points of individuals with maximal fitness found during evolution
    pub fn dump_fittest(self: *NoveltyArchive, path: []const u8) !void {
        if (self.fittest_items.items.len == 0) return error.NoNovelItems;

        const dir_path = std.fs.path.dirname(path);
        const file_name = std.fs.path.basename(path);
        var file_dir: std.fs.Dir = undefined;
        if (dir_path != null) {
            file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
        } else {
            file_dir = std.fs.cwd();
        }
        var output_file = try file_dir.createFile(file_name, .{});
        defer output_file.close();
        try self.dump_novelty_items(self.fittest_items.items, output_file.writer());
    }

    fn dump_novelty_items(_: *NoveltyArchive, items: []*NoveltyItem, writer: anytype) !void {
        try json.toWriter(null, items, writer);
    }
};

// test utility functions

fn fill_organism_data(allocator: std.mem.Allocator, org: *Organism, novelty: f64) !void {
    var nov_item = try allocator.create(NoveltyItem);
    var nov_data = std.ArrayList(f64).init(allocator);
    try nov_data.append(0.1);
    nov_item.* = .{
        .allocator = allocator,
        .generation = org.generation,
        .fitness = org.fitness,
        .novelty = novelty,
        .data = nov_data,
    };
    org.data = nov_item;
}

fn square_metric(x: *NoveltyItem, y: *NoveltyItem) f64 {
    return (x.fitness - y.fitness) * (x.fitness - y.fitness);
}

fn create_rand_population(allocator: std.mem.Allocator, rand: std.rand.Random, in: usize, out: usize, max_hidden: usize, link_prob: f64, opts: *Options) !*Population {
    var pop = try Population.init_random(allocator, rand, in, out, max_hidden, false, link_prob, opts);
    for (pop.organisms.items, 0..) |org, i| {
        var float_val = 0.1 * (1 + @as(f64, @floatFromInt(i)));
        org.fitness = float_val;
        try fill_organism_data(allocator, org, float_val);
    }
    return pop;
}

// NoveltyArchive Unit tests
test "NoveltyArchive update fittest with Organism" {
    var allocator = std.testing.allocator;
    var opts = try NoveltyArchiveOptions.init(allocator);
    var archive = try NoveltyArchive.init(allocator, 1, &square_metric, opts);
    defer archive.deinit();

    var gen = try Genome.read_from_file(allocator, "src/ns/test_data/initgenome");

    // used to free allocated organisms
    var orgs_list = std.ArrayList(*Organism).init(allocator);
    defer orgs_list.deinit();
    defer for (orgs_list.items) |organism| {
        if (organism.data != null) {
            var org_item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(organism.data.?)));
            org_item.deinit();
        }
        organism.deinit();
    };

    var org = try Organism.init(allocator, 0.1, gen, 1);
    try orgs_list.append(org);
    try fill_organism_data(allocator, org, 0.0);

    try archive.update_fittest_with_organism(allocator, org);
    try std.testing.expect(archive.fittest_items.items.len == 1);

    var idx: usize = 2;
    while (idx <= opts.fittest_allowed_size) : (idx += 1) {
        var gen_copy = try gen.duplicate(allocator, gen.id);
        var new_org = try Organism.init(allocator, 0.1 * @as(f64, @floatFromInt(idx)), gen_copy, 1);
        try orgs_list.append(new_org);
        try fill_organism_data(allocator, new_org, 0.0);
        try archive.update_fittest_with_organism(allocator, new_org);
    }

    for (archive.fittest_items.items, 0..) |item, i| {
        var expected = @as(f64, @floatFromInt(opts.fittest_allowed_size - i)) * 0.1;
        try std.testing.expect(item.fitness == expected);
    }

    // test update over allowed size
    var fitness: f64 = 0.6;
    var gen_copy = try gen.duplicate(allocator, gen.id);
    var new_org = try Organism.init(allocator, fitness, gen_copy, 1);
    try orgs_list.append(new_org);
    try fill_organism_data(allocator, new_org, 0.0);
    try archive.update_fittest_with_organism(allocator, new_org);
    try std.testing.expect(archive.fittest_items.items.len == opts.fittest_allowed_size);
    try std.testing.expect(archive.fittest_items.items[0].fitness == fitness);
}

test "NoveltyArchive add NoveltyItem" {
    var allocator = std.testing.allocator;
    var opts = try NoveltyArchiveOptions.init(allocator);
    var archive = try NoveltyArchive.init(allocator, 1, &square_metric, opts);
    defer archive.deinit();

    var gen = try Genome.read_from_file(allocator, "src/ns/test_data/initgenome");
    var org = try Organism.init(allocator, 0.1, gen, 1);
    defer org.deinit();
    try fill_organism_data(allocator, org, 0.0);

    // test add novelty item
    var item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
    defer item.deinit();
    try archive.add_novelty_item(item);

    try std.testing.expect(archive.novel_items.items.len == 1);
    try std.testing.expect(archive.novel_items.items[0].added);
    try std.testing.expect(archive.novel_items.items[0].generation == archive.generation);
    try std.testing.expect(archive.items_added_in_generation == 1);
}

test "NoveltyArchive evaluate individual" {
    var allocator = std.testing.allocator;
    var neat_opts = Options{
        .compat_threshold = 0.5,
        .pop_size = 10,
    };
    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var pop = try create_rand_population(allocator, rand, 3, 2, 5, 0.5, &neat_opts);
    defer pop.deinit();
    defer for (pop.organisms.items) |org| {
        if (org.data != null) {
            var org_item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
            org_item.deinit();
        }
    };

    var opts = try NoveltyArchiveOptions.init(allocator);
    var archive = try NoveltyArchive.init(allocator, 1, &square_metric, opts);
    defer archive.deinit();
    archive.generation = 2;

    // test evaluate only in archive
    var org = pop.organisms.items[0];
    try archive.evaluate_individual_novelty(allocator, org, pop, false);
    try std.testing.expect(archive.novel_items.items.len == 1);
    try std.testing.expect(archive.novel_items.items[0].added);

    // check that data object properly filled
    var item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
    try std.testing.expect(item.added);
    try std.testing.expect(item.generation == archive.generation);

    // test evaluate in population as well
    try archive.evaluate_individual_novelty(allocator, org, pop, true);
    try std.testing.expect(archive.novel_items.items.len == 1);
    try std.testing.expect(org.fitness != 0.1);
}

test "NoveltyArchive evaluate Population" {
    var allocator = std.testing.allocator;
    var neat_opts = Options{
        .compat_threshold = 0.5,
        .pop_size = 10,
    };
    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var pop = try create_rand_population(allocator, rand, 3, 2, 5, 0.5, &neat_opts);
    defer pop.deinit();
    defer for (pop.organisms.items) |org| {
        if (org.data != null) {
            var org_item: *NoveltyItem = @as(*NoveltyItem, @ptrCast(@alignCast(org.data.?)));
            org_item.deinit();
        }
    };

    var opts = try NoveltyArchiveOptions.init(allocator);
    var archive = try NoveltyArchive.init(allocator, 0.1, &square_metric, opts);
    defer archive.deinit();
    archive.generation = 2;

    // test update fitness scores
    try archive.evaluate_population_novelty(allocator, pop, true);
    for (pop.organisms.items, 0..) |org, i| {
        var not_expected: f64 = 0.1 * (1 + @as(f64, @floatFromInt(i)));
        try std.testing.expect(org.fitness != not_expected);
    }

    // test add to archive
    try archive.evaluate_population_novelty(allocator, pop, false);
    try std.testing.expect(archive.novel_items.items.len == 3);
}
