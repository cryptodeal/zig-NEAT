const std = @import("std");
const novelty_item = @import("novelty_item.zig");

pub const NoveltyArchive = @import("novelty_archive.zig").NoveltyArchive;
pub const NoveltyItem = novelty_item.NoveltyItem;
pub const ItemsDistance = novelty_item.ItemsDistance;

/// how many nearest neighbors to consider for calculating novelty score?
pub const knn_novelty_score: usize = 15;

/// the maximal allowed size for fittest items list
pub const fittest_allowed_size: usize = 5;

/// the minimal number of seed novelty items to start from
pub const archive_seed_amount: usize = 1;

pub const NoveltyMetric = fn (*NoveltyItem, *NoveltyItem) f64;

/// NoveltyArchiveOptions defines options to be used by NoveltyArchive
pub const NoveltyArchiveOptions = struct {
    /// how many nearest neighbors to consider for calculating novelty score, i.e., for how many
    /// neighbors to look at for N-nearest neighbor distance novelty
    knn_novelty_score: usize = knn_novelty_score,
    /// the maximal allowed size for fittest items list
    fittest_allowed_size: usize = fittest_allowed_size,
    /// the minimal number of seed novelty items to start from
    archive_seed_amount: usize = archive_seed_amount,
};

test {
    std.testing.refAllDecls(@This());
}
