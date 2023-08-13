const std = @import("std");
const NoveltyArchiveOptions = @import("./common.zig").NoveltyArchiveOptions;

/// NoveltyItem is the data holder for novel item's genome and phenotype
pub const NoveltyItem = struct {
    /// Flag indicating whether item was added to archive.
    added: bool = false,
    /// Generation when item was added to the archive.
    generation: usize = undefined,
    /// The id of the associated Organism.
    individual_id: u64 = undefined,
    /// Fitness score of the associated Organism.
    fitness: f64 = undefined,
    /// The novelty score of the item.
    novelty: f64 = undefined,
    /// The item's age.
    age: usize = undefined,
    /// The data associated with the item.
    data: std.ArrayList(f64),
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new NoveltyItem.
    pub fn init(allocator: std.mem.Allocator) !*NoveltyItem {
        var self = try allocator.create(NoveltyItem);
        self.* = .{
            .allocator = allocator,
            .data = std.ArrayList(f64).init(allocator),
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *NoveltyItem) void {
        self.data.deinit();
        self.allocator.destroy(self);
    }

    /// Formats NoveltyItem for printing to writer.
    pub fn format(value: NoveltyItem, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Novelty: {d:.2} Fitness: {d:.6} Generation: {d} Individual: {d}\n", .{ value.novelty, value.fitness, value.generation, value.individual_id });
        try writer.print("\tPoint: ", .{});
        for (value.data.items) |d| {
            try writer.print(" {d:.3}", .{d});
        }
    }

    pub const @"getty.sb" = struct {
        pub const attributes = .{
            .allocator = .{ .skip = true },
        };
    };
};

/// ItemsDistance holds distance between two items
pub const ItemsDistance = struct {
    /// Distance between two NoveltyItems.
    distance: f64,
    /// The NoveltyItem from which distance was calculated.
    from: *NoveltyItem,
    /// The NoveltyItem to which distance was calculated.
    to: *NoveltyItem,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new ItemsDistance.
    pub fn init(allocator: std.mem.Allocator, from: *NoveltyItem, to: *NoveltyItem, distance: f64) !*ItemsDistance {
        var self = try allocator.create(ItemsDistance);
        self.* = .{
            .allocator = allocator,
            .from = from,
            .to = to,
            .distance = distance,
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *ItemsDistance) void {
        self.allocator.destroy(self);
    }
};

/// Used to sort []ItemsDistance by distance.
pub fn itemsDistanceComparison(context: void, a: *ItemsDistance, b: *ItemsDistance) bool {
    _ = context;
    return a.distance < b.distance;
}

/// Used to sort []*NoveltyItem by fitness; if fitness between two items is equal, falls back to sort by novelty score.
pub fn noveltyItemComparison(context: void, a: *NoveltyItem, b: *NoveltyItem) bool {
    _ = context;
    if (a.fitness < b.fitness) {
        return true;
    } else if (a.fitness == b.fitness and a.novelty < b.novelty) {
        return true; // less novel is less
    }

    return false;
}

test "NoveltyItem format" {
    var allocator = std.testing.allocator;
    var res_str = std.ArrayList(u8).init(allocator);
    defer res_str.deinit();

    var item = try NoveltyItem.init(allocator);
    defer item.deinit();
    try item.data.append(100.1);
    try item.data.append(123.9);
    item.generation = 1;
    item.individual_id = 10;
    item.fitness = 0.5;
    item.novelty = 25.35;
    item.age = 2;

    try item.format("", .{}, res_str.writer());
    var res: []const u8 = try res_str.toOwnedSlice();
    defer allocator.free(res);
    try std.testing.expect(std.mem.eql(u8, res, "Novelty: 25.35 Fitness: 0.500000 Generation: 1 Individual: 10\n\tPoint:  100.100 123.900"));
}
