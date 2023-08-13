const std = @import("std");

/// DetectionSide the side of retina where VisualObject is valid to be detected
pub const DetectionSide = enum {
    RightSide,
    LeftSide,
    BothSides,
};

/// Environment holds the dataset and evaluation methods for the modular retina experiment
pub const Environment = struct {
    // the data set of visual objects to be detected
    visual_objects: []*VisualObject,
    // the size of input data array
    input_size: usize,

    allocator: std.mem.Allocator,

    /// Creates a new Retina Environment with a dataset of all possible Visual Object with specified
    /// number of inputs to be acquired from provided objects.
    pub fn init(allocator: std.mem.Allocator, data_set: []*VisualObject, input_size: usize) !*Environment {
        // check that provided visual objects has data points equal to the inputSize
        for (data_set) |o| {
            if (o.data.len != input_size) {
                std.debug.print("all visual objects expected to have {d} data points, but found {d} at {any}\n", .{ input_size, o.data.len, o });
                return error.InvalidVisualObjectDataLength;
            }
        }
        var self: *Environment = try allocator.create(Environment);
        self.* = .{
            .allocator = allocator,
            .visual_objects = data_set,
            .input_size = input_size,
        };
        return self;
    }

    pub fn deinit(self: *Environment) void {
        // TODO: for (self.visual_objects) |o| o.deinit();
        // TODO: self.allocator.free(self.visual_objects);
        self.allocator.destroy(self);
    }
};

/// VisualObject represents a left, right, or both, object classified by retina
pub const VisualObject = struct {
    /// the side(s) of retina where this visual object accepted as valid
    side: DetectionSide,
    /// the configuration string
    config: []const u8,

    /// Inner computed values from visual objects configuration parsing
    data: []f64, // the visual object is rectangular, it can be encoded as 1D array

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, side: DetectionSide, config: []const u8) !*VisualObject {
        var self: *VisualObject = try allocator.create(VisualObject);
        // Setup visual object data multi-array from config string
        self.* = .{
            .allocator = allocator,
            .side = side,
            .config = config,
            .data = try parseVisualObjectConfig(allocator, config),
        };
        return self;
    }

    pub fn deinit(self: *VisualObject) void {
        self.allocator.free(self.data);
        self.allocator.destroy(self);
    }

    pub fn format(value: VisualObject, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("{s}\n{s}", .{ @tagName(value.side), value.config });
    }
};

/// parses config semantically in the format
/// (config = "x1 x2 \n x3 x4") to [ xf1, xf2, xf3, xf4 ]  where if xi == "o" => xfi = 1.0
fn parseVisualObjectConfig(allocator: std.mem.Allocator, config: []const u8) ![]f64 {
    var data = std.ArrayList(f64).init(allocator);
    var new_line_iterator = std.mem.split(u8, config, "\n");
    while (new_line_iterator.next()) |line| {
        var chars_iterator = std.mem.split(u8, line, " ");
        while (chars_iterator.next()) |char| {
            if (std.mem.eql(u8, char, "o")) {
                // pixel is ON
                try data.append(1);
            } else if (std.mem.eql(u8, char, ".")) {
                // pixel is OFF
                try data.append(0);
            } else {
                std.debug.print("unsupported configuration character [{s}]\n", .{char});
                return error.UnsupportedConfigurationCharacter;
            }
        }
    }
    return data.toOwnedSlice();
}

test "parse VisualObject config" {
    const allocator = std.testing.allocator;
    const Resource = struct { config: []const u8, expected: [4]f64 };
    const resources = [_]Resource{
        .{ .config = ". o\n. o", .expected = [4]f64{ 0, 1, 0, 1 } },
        .{ .config = "o .\n. o", .expected = [4]f64{ 1, 0, 0, 1 } },
        .{ .config = "o .\n. .", .expected = [4]f64{ 1, 0, 0, 0 } },
        .{ .config = ". .\n. .", .expected = [4]f64{ 0, 0, 0, 0 } },
    };
    for (resources) |res| {
        var data = try parseVisualObjectConfig(allocator, res.config);
        defer allocator.free(data);
        try std.testing.expectEqualSlices(f64, &res.expected, data);
    }
}
