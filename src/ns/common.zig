const std = @import("std");
const novelty_item = @import("novelty_item.zig");

pub const NoveltyArchive = @import("novelty_archive.zig").NoveltyArchive;
pub const NoveltyItem = novelty_item.NoveltyItem;
pub const ItemsDistance = novelty_item.ItemsDistance;

/// How many nearest neighbors to consider for calculating novelty score?
pub const knn_novelty_score: usize = 15;

/// The maximal allowed size for fittest items list
pub const fittest_allowed_size: usize = 5;

/// The minimal number of seed novelty items to start from
pub const archive_seed_amount: usize = 1;

/// NoveltyMetric is the type describing the function used to calculate
/// novelty score between two novelty items.
pub const NoveltyMetric = fn (*NoveltyItem, *NoveltyItem) f64;

/// NoveltyArchiveOptions defines options to be used by NoveltyArchive.
pub const NoveltyArchiveOptions = struct {
    /// How many nearest neighbors to consider for calculating novelty score, i.e., for how many
    /// neighbors to look at for N-nearest neighbor distance novelty.
    knn_novelty_score: usize = knn_novelty_score,
    /// The maximal allowed size for fittest items list.
    fittest_allowed_size: usize = fittest_allowed_size,
    /// The minimal number of seed novelty items to start from
    archive_seed_amount: usize = archive_seed_amount,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator = undefined,

    /// Initialize NoveltyArchiveOptions with default values.
    pub fn init(allocator: std.mem.Allocator) !*NoveltyArchiveOptions {
        var self = try allocator.create(NoveltyArchiveOptions);
        self.* = .{ .allocator = allocator };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *NoveltyArchiveOptions) void {
        self.allocator.destroy(self);
    }
};

test {
    std.testing.refAllDecls(@This());
}
