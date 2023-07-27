const std = @import("std");
const graph = @import("../graph/graph.zig");
const Network = @import("network.zig").Network;

const GraphNode = graph.GraphNode;

pub fn print_all_activation_depth_paths(n: *Network, writer: anytype) !void {
    _ = writer;
    _ = n;
    // TODO: implement
}

/// prints the given paths into specified writer
pub fn print_path(writer: anytype, paths: ?std.ArrayList(std.ArrayList(*GraphNode(i64, {})))) !void {
    if (paths == null) {
        return error.PathsAreEmpty;
    }
    for (paths.items) |p| {
        var l = p.items.len;
        for (p.items, 0..) |n, i| {
            if (i < l - 1) {
                try writer.print("{d} -> ", .{n.id});
            } else {
                try writer.print("{d}", .{n.id});
            }
        }
    }
}
