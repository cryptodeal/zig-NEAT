const std = @import("std");
const normalizedFloatHash = @import("cppn.zig").normalizedFloatHash;

/// PointFHash defines a hashable value for a given PointF instance.
pub const PointFHash = packed struct {
    x_hash: u64,
    y_hash: u64,
};

/// PointF defines a point with `f64` precision coordinates.
pub const PointF = struct {
    /// The X coordinate of this point.
    x: f64,
    /// The Y coordinate of this point.
    y: f64,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new PointF with the given coordinates.
    pub fn init(allocator: std.mem.Allocator, x: f64, y: f64) !*PointF {
        var self = try allocator.create(PointF);
        self.* = .{
            .allocator = allocator,
            .x = x,
            .y = y,
        };
        return self;
    }

    /// Returns the hash value for this PointF.
    pub fn key(self: *PointF) PointFHash {
        return .{ .x_hash = self.hashFloat(self.x), .y_hash = self.hashFloat(self.y) };
    }

    fn hashFloat(_: *PointF, v: f64) u64 {
        var hasher = std.hash.Wyhash.init(0);
        normalizedFloatHash(&hasher, v);
        return hasher.final();
    }

    /// Frees all associated memory.
    pub fn deinit(self: *PointF) void {
        self.allocator.destroy(self);
    }

    /// Formats PointF for printing to writer.
    pub fn format(value: PointF, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("({d}, {d})", .{ value.x, value.y });
    }
};

/// QuadPoint defines the quad-point in the 4 dimensional hypercube.
pub const QuadPoint = struct {
    /// associated x1 coordinate.
    x1: f64,
    /// associated x2 coordinate.
    x2: f64,
    /// associated y1 coordinate.
    y1: f64,
    /// associated y2 coordinate.
    y2: f64,
    /// The value for this point.
    cppn_out: []f64,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new QuadPoint with the given coordinates and node's CPPN outputs.
    pub fn init(allocator: std.mem.Allocator, x1: f64, y1: f64, x2: f64, y2: f64, node: *QuadNode) !*QuadPoint {
        var self = try allocator.create(QuadPoint);
        var outs = try allocator.alloc(f64, node.cppn_out.len);
        @memcpy(outs, node.cppn_out);
        self.* = .{
            .allocator = allocator,
            .x1 = x1,
            .y1 = y1,
            .x2 = x2,
            .y2 = y2,
            .cppn_out = outs,
        };
        return self;
    }

    /// Frees all associated memory.
    pub fn deinit(self: *QuadPoint) void {
        self.allocator.free(self.cppn_out);
        self.allocator.destroy(self);
    }

    /// Returns the weight for this QuadPoint (weight derived from Point's CPPN outputs).
    pub fn weight(self: *QuadPoint) f64 {
        return self.cppn_out[0];
    }

    /// Formats QuadPoint for printing to writer.
    pub fn format(value: QuadPoint, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("(({d}, {d}),({d}, {d})) = {d}", .{ value.x1, value.y1, value.x2, value.y2, value.value });
    }
};

/// QuadNode defines a quad-tree node to model 4 dimensional hypercube.
pub const QuadNode = struct {
    /// The X coordinates of center of this quad-tree node's square.
    x: f64,
    /// The Y coordinates of center of this quad-tree node's square.
    y: f64,
    /// The width of this quad-tree node's square.
    width: f64,
    // The CPPN outputs for this node.
    cppn_out: []f64,
    /// The level of this node in the quad-tree.
    level: usize,
    /// The children of this node.
    nodes: std.ArrayList(*QuadNode),
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new QuadNode with the given coordinates, width, and level.
    pub fn init(allocator: std.mem.Allocator, x: f64, y: f64, width: f64, level: usize) !*QuadNode {
        var self = try allocator.create(QuadNode);
        var out = try allocator.alloc(f64, 1);
        out[0] = 0;
        self.* = .{
            .allocator = allocator,
            .x = x,
            .y = y,
            .width = width,
            .cppn_out = out,
            .level = level,
            .nodes = std.ArrayList(*QuadNode).init(allocator),
        };
        return self;
    }

    /// Returns the weight for this QuadNode (weight derived from QuadNode's CPPN outputs).
    pub fn weight(self: *QuadNode) f64 {
        return self.cppn_out[0];
    }

    /// Frees all associated memory.
    pub fn deinit(self: *QuadNode) void {
        self.allocator.free(self.cppn_out);
        for (self.nodes.items) |n| n.deinit();
        self.nodes.deinit();
        self.allocator.destroy(self);
    }

    /// Formats QuadNode for printing to writer.
    pub fn format(value: QuadNode, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("(({d}, {d}), {d}) = {any} at {d}", .{ value.x, value.y, value.width, value.cppn_out, value.level });
    }
};
