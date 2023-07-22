const std = @import("std");

pub fn create_out_dir_for_trial(writer: anytype, out_dir: []const u8, trial_id: usize) !void {
    try writer.print("{s}/{d}", .{ out_dir, trial_id });
}
