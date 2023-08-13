const std = @import("std");
const quad_tree = @import("quad_tree.zig");
const net_common = @import("../network/common.zig");

const NodeNeuronType = net_common.NodeNeuronType;
const PointFHash = quad_tree.PointFHash;
const PointF = quad_tree.PointF;
const assert = std.debug.assert;

pub const SubstrateLayout = struct {
    const Self = @This();
    // The type erased pointer to the GenerationEvaluator implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        nodePosition: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, index: usize, n_type: NodeNeuronType) anyerror!*PointF,
        bias_count: *const fn (ctx: *anyopaque) usize,
        inputCount: *const fn (ctx: *anyopaque) usize,
        hiddenCount: *const fn (ctx: *anyopaque) usize,
        outputCount: *const fn (ctx: *anyopaque) usize,
        deinit: *const fn (ctx: *anyopaque) void,
    };

    /// Returns coordinates of the neuron with specified index [0; count) and type
    pub fn nodePosition(self: Self, allocator: std.mem.Allocator, index: usize, n_type: NodeNeuronType) anyerror!*PointF {
        return try self.vtable.nodePosition(self.ptr, allocator, index, n_type);
    }

    /// Return number of BIAS neurons in the layout
    pub fn bias_count(self: Self) usize {
        return self.vtable.bias_count(self.ptr);
    }

    /// Returns number of INPUT neurons in the layout
    pub fn inputCount(self: Self) usize {
        return self.vtable.inputCount(self.ptr);
    }

    /// Returns number of HIDDEN neurons in the layout
    pub fn hiddenCount(self: Self) usize {
        return self.vtable.hiddenCount(self.ptr);
    }

    /// Returns number of OUTPUT neurons in the layout
    pub fn outputCount(self: Self) usize {
        return self.vtable.outputCount(self.ptr);
    }

    /// Frees any memory allocated by the layout implementation
    pub fn deinit(self: Self) void {
        return self.vtable.deinit(self.ptr);
    }

    pub fn init(es_layout: anytype) Self {
        const Ptr = @TypeOf(es_layout);
        const PtrInfo = @typeInfo(Ptr);
        assert(PtrInfo == .Pointer); // Must be a pointer
        assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
        assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
        const impl = struct {
            fn nodePosition(ctx: *anyopaque, allocator: std.mem.Allocator, index: usize, n_type: NodeNeuronType) !*PointF {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.nodePosition(allocator, index, n_type);
            }

            fn bias_count(ctx: *anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.bias_count();
            }

            fn inputCount(ctx: *anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.inputCount();
            }

            fn hiddenCount(ctx: *anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.hiddenCount();
            }

            fn outputCount(ctx: *anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.outputCount();
            }

            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.deinit();
            }
        };
        return .{
            .ptr = es_layout,
            .vtable = &.{
                .nodePosition = impl.nodePosition,
                .bias_count = impl.bias_count,
                .inputCount = impl.inputCount,
                .hiddenCount = impl.hiddenCount,
                .outputCount = impl.outputCount,
                .deinit = impl.deinit,
            },
        };
    }
};

/// Defines grid substrate layout
pub const GridSubstrateLayout = struct {
    /// The number of bias nodes encoded in this substrate
    num_bias: usize,
    /// The number of input nodes encoded in this substrate
    num_input: usize,
    /// The number of hidden nodes encoded in this substrate
    num_hidden: usize,
    /// The number of output nodes encoded in this substrate
    num_output: usize,

    /// The input coordinates increment
    input_delta: f64 = 0,
    /// The hidden coordinates increment
    hidden_delta: f64 = 0,
    /// The output coordinates increment
    output_delta: f64 = 0,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_bias: usize, num_input: usize, num_output: usize, num_hidden: usize) !*GridSubstrateLayout {
        var self = try allocator.create(GridSubstrateLayout);
        self.* = .{
            .num_bias = num_bias,
            .num_input = num_input,
            .num_hidden = num_hidden,
            .num_output = num_output,
            .allocator = allocator,
        };
        if (num_input != 0) {
            self.input_delta = 2 / @as(f64, @floatFromInt(num_input));
        }
        if (num_hidden != 0) {
            self.hidden_delta = 2 / @as(f64, @floatFromInt(num_hidden));
        }
        if (num_output != 0) {
            self.output_delta = 2 / @as(f64, @floatFromInt(num_output));
        }
        return self;
    }

    pub fn deinit(self: *GridSubstrateLayout) void {
        self.allocator.destroy(self);
    }

    pub fn nodePosition(self: *GridSubstrateLayout, allocator: std.mem.Allocator, index: usize, n_type: NodeNeuronType) !*PointF {
        var point = try PointF.init(allocator, 0, 0);
        errdefer point.deinit();
        var delta: f64 = undefined;
        var count: usize = undefined;
        switch (n_type) {
            .BiasNeuron => {
                if (index < self.num_bias) {
                    return point; // BIAS always located at (0, 0)
                } else {
                    std.debug.print("the BIAS index is out of range\n", .{});
                    return error.BiasIndexOutOfRange;
                }
            },
            .HiddenNeuron => {
                delta = self.hidden_delta;
                count = self.num_hidden;
            },
            .InputNeuron => {
                delta = self.input_delta;
                count = self.num_input;
                point.y = -1;
            },
            .OutputNeuron => {
                delta = self.output_delta;
                count = self.num_output;
                point.y = 1;
            },
        }

        if (index >= count) {
            std.debug.print("neuron index is out of range\n", .{});
            return error.NeuronIndexOutOfRange;
        }
        // calculate X position
        point.x = -1 + delta / 2; // the initial position with half delta shift
        point.x += @as(f64, @floatFromInt(index)) * delta;
        return point;
    }

    pub fn bias_count(self: *GridSubstrateLayout) usize {
        return self.num_bias;
    }

    pub fn inputCount(self: *GridSubstrateLayout) usize {
        return self.num_input;
    }

    pub fn hiddenCount(self: *GridSubstrateLayout) usize {
        return self.num_hidden;
    }

    pub fn outputCount(self: *GridSubstrateLayout) usize {
        return self.num_output;
    }
};

// test utils/unit tests

fn checkNeuronLayoutPositions(allocator: std.mem.Allocator, positions: []f64, n_type: NodeNeuronType, layout: SubstrateLayout) !void {
    const count = positions.len / 2;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        var pos = try layout.nodePosition(allocator, i, n_type);
        defer pos.deinit();
        try std.testing.expect(pos.x == positions[i * 2] and pos.y == positions[i * 2 + 1]);
    }
}

test "SubstrateLayout node position" {
    const allocator = std.testing.allocator;
    const bias_count: usize = 1;
    const input_count: usize = 4;
    const hidden_count: usize = 2;
    const output_count: usize = 2;

    var layout = SubstrateLayout.init(try GridSubstrateLayout.init(allocator, bias_count, input_count, output_count, hidden_count));
    defer layout.deinit();

    // check BIAS
    var d1 = [_]f64{ 0, 0 };
    try checkNeuronLayoutPositions(allocator, &d1, .BiasNeuron, layout);
    try std.testing.expectError(error.BiasIndexOutOfRange, layout.nodePosition(allocator, 1, .BiasNeuron));

    // check INPUT
    var d2 = [_]f64{ -0.75, -1, -0.25, -1, 0.25, -1, 0.75, -1 };
    try checkNeuronLayoutPositions(allocator, &d2, .InputNeuron, layout);
    try std.testing.expectError(error.NeuronIndexOutOfRange, layout.nodePosition(allocator, input_count, .InputNeuron));

    // check HIDDEN
    var d3 = [_]f64{ -0.5, 0, 0.5, 0 };
    try checkNeuronLayoutPositions(allocator, &d3, .HiddenNeuron, layout);
    try std.testing.expectError(error.NeuronIndexOutOfRange, layout.nodePosition(allocator, hidden_count, .HiddenNeuron));

    // check OUTPUT
    var d4 = [_]f64{ -0.5, 1, 0.5, 1 };
    try checkNeuronLayoutPositions(allocator, &d4, .OutputNeuron, layout);
    try std.testing.expectError(error.NeuronIndexOutOfRange, layout.nodePosition(allocator, output_count, .OutputNeuron));
}
