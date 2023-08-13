const std = @import("std");

/// Creates the output directory for specific trial of the experiment using standard name.
pub fn createOutDirForTrial(writer: anytype, out_dir: []const u8, trial_id: usize) !void {
    try writer.print("{s}/{d}", .{ out_dir, trial_id });
}
