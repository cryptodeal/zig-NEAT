const std = @import("std");
const quad_tree = @import("quad_tree.zig");
const net_common = @import("../network/common.zig");

const NodeNeuronType = net_common.NodeNeuronType;
const PointFHash = quad_tree.PointFHash;
const PointF = quad_tree.PointF;
const assert = std.debug.assert;

pub const EvolvableSubstrateLayout = struct {
    const Self = @This();
    // The type erased pointer to the GenerationEvaluator implementation
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        node_position: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, index: usize, n_type: NodeNeuronType) anyerror!*PointF,
        add_hidden_node: *const fn (ctx: *anyopaque, position: *PointF) anyerror!usize,
        index_of_hidden: *const fn (ctx: *anyopaque, position: *PointF) anyerror!usize,
        input_count: *const fn (ctx: *anyopaque) usize,
        hidden_count: *const fn (ctx: *anyopaque) usize,
        output_count: *const fn (ctx: *anyopaque) usize,
        deinit: *const fn (ctx: *anyopaque) void,
    };

    /// Returns coordinates of the neuron with specified index [0; count) and type
    pub fn node_position(self: Self, allocator: std.mem.Allocator, index: usize, n_type: NodeNeuronType) anyerror!*PointF {
        return try self.vtable.node_position(self.ptr, allocator, index, n_type);
    }

    /// Adds new hidden node to the substrate
    /// Returns the index of added hidden neuron or error if failed.
    pub fn add_hidden_node(self: Self, position: *PointF) anyerror!usize {
        return try self.vtable.add_hidden_node(self.ptr, position);
    }

    /// Returns index of hidden node at specified position or error if not found
    pub fn index_of_hidden(self: Self, position: *PointF) !usize {
        return self.vtable.index_of_hidden(self.ptr, position);
    }

    /// Returns number of INPUT neurons in the layout
    pub fn input_count(self: Self) usize {
        return self.vtable.input_count(self.ptr);
    }

    /// Returns number of HIDDEN neurons in the layout
    pub fn hidden_count(self: Self) usize {
        return self.vtable.hidden_count(self.ptr);
    }

    /// Returns number of OUTPUT neurons in the layout
    pub fn output_count(self: Self) usize {
        return self.vtable.output_count(self.ptr);
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
            fn node_position(ctx: *anyopaque, allocator: std.mem.Allocator, index: usize, n_type: NodeNeuronType) !*PointF {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.node_position(allocator, index, n_type);
            }

            fn add_hidden_node(ctx: *anyopaque, position: *PointF) !usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.add_hidden_node(position);
            }

            fn index_of_hidden(ctx: *anyopaque, position: *PointF) !usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.index_of_hidden(position);
            }

            fn input_count(ctx: *anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.input_count();
            }

            fn hidden_count(ctx: *anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.hidden_count();
            }

            fn output_count(ctx: *anyopaque) usize {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.output_count();
            }

            fn deinit(ctx: *anyopaque) void {
                const self: Ptr = @ptrCast(@alignCast(ctx));
                return self.deinit();
            }
        };
        return .{
            .ptr = es_layout,
            .vtable = &.{
                .node_position = impl.node_position,
                .add_hidden_node = impl.add_hidden_node,
                .index_of_hidden = impl.index_of_hidden,
                .input_count = impl.input_count,
                .hidden_count = impl.hidden_count,
                .output_count = impl.output_count,
                .deinit = impl.deinit,
            },
        };
    }
};

pub const MappedEvolvableSubstrateLayout = struct {
    /// The map to hold binding between hidden node and its index for fast search
    h_nodes_map: std.AutoHashMap(PointFHash, usize), // stores pointer as usize and index
    /// The list of all known hidden nodes in specific order
    h_nodes_list: std.ArrayList(*PointF),
    /// The number of input nodes encoded in this substrate
    in_count: usize,
    /// The number of output nodes encoded in this substrate
    out_count: usize,
    /// The input coordinates increment
    input_delta: f64,
    // The output coordinates increment
    output_delta: f64,

    allocator: std.mem.Allocator,

    /// Creates new instance with given input and output neurons count
    pub fn init(allocator: std.mem.Allocator, in_count: usize, out_count: usize) !*MappedEvolvableSubstrateLayout {
        if (in_count == 0) {
            std.debug.print("the number of input neurons can not be ZERO\n", .{});
            return error.InvalidInputCount;
        }
        if (out_count == 0) {
            std.debug.print("the number of output neurons can not be ZERO\n", .{});
            return error.InvalidOutputCount;
        }
        var l = try allocator.create(MappedEvolvableSubstrateLayout);
        l.* = .{
            .allocator = allocator,
            .h_nodes_map = std.AutoHashMap(PointFHash, usize).init(allocator),
            .h_nodes_list = std.ArrayList(*PointF).init(allocator),
            .in_count = in_count,
            .out_count = out_count,
            .input_delta = 2 / @as(f64, @floatFromInt(in_count)),
            .output_delta = 2 / @as(f64, @floatFromInt(out_count)),
        };
        return l;
    }

    pub fn deinit(self: *MappedEvolvableSubstrateLayout) void {
        self.h_nodes_map.deinit();
        for (self.h_nodes_list.items) |n| n.deinit();
        self.h_nodes_list.deinit();
        self.allocator.destroy(self);
    }

    pub fn node_position(self: *MappedEvolvableSubstrateLayout, allocator: std.mem.Allocator, index: usize, n_type: NodeNeuronType) !*PointF {
        var y: f64 = 0;
        var delta: f64 = 0;
        var count: usize = 0;

        switch (n_type) {
            .BiasNeuron => {
                std.debug.print("the BIAS neurons is not supported by Evolvable Substrate\n", .{});
                return error.BiasNeuronsNotSupported;
            },
            .HiddenNeuron => count = self.h_nodes_list.items.len,
            .InputNeuron => {
                delta = self.input_delta;
                count = self.in_count;
                y = -1;
            },
            .OutputNeuron => {
                delta = self.output_delta;
                count = self.out_count;
                y = 1;
            },
        }

        if (index >= count) {
            std.debug.print("neuron index is out of range\n", .{});
            return error.NeuronIndexOutOfRange;
        } else if (n_type == .HiddenNeuron) {
            // return stored hidden neuron position
            return self.h_nodes_list.items[index];
        }

        var point = try PointF.init(allocator, 0, y);

        // calculate X position
        point.x = -1 + delta / 2; // the initial position with half delta shift
        point.x += @as(f64, @floatFromInt(index)) * delta;
        return point;
    }

    pub fn add_hidden_node(self: *MappedEvolvableSubstrateLayout, position: *PointF) !usize {
        _ = self.index_of_hidden(position) catch {
            // add to the list and map it
            try self.h_nodes_list.append(position);
            var index = self.h_nodes_list.items.len - 1;
            try self.h_nodes_map.put(position.key(), index);
            return index;
        };

        std.debug.print("hidden node already exists at the position: {any}\n", .{position});
        return error.HiddenNodeAlreadyExists;
    }

    pub fn index_of_hidden(self: *MappedEvolvableSubstrateLayout, position: *PointF) !usize {
        var index: ?usize = self.h_nodes_map.get(position.key());
        if (index == null) return error.HiddenNodeNotFound;
        return index.?;
    }

    pub fn bias_count(_: *MappedEvolvableSubstrateLayout) usize {
        // No BIAS nodes
        return 0;
    }

    pub fn input_count(self: *MappedEvolvableSubstrateLayout) usize {
        return self.in_count;
    }

    pub fn hidden_count(self: *MappedEvolvableSubstrateLayout) usize {
        return self.h_nodes_list.items.len;
    }

    pub fn output_count(self: *MappedEvolvableSubstrateLayout) usize {
        return self.out_count;
    }

    pub fn format(value: MappedEvolvableSubstrateLayout, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("MappedEvolvableSubstrateLayout:\n\tINPT: {d}\n\tHIDN: {d}\n\tOUTP: {d}", .{ value.input_count(), value.hidden_count(), value.output_count() });
    }
};

pub fn check_neuron_layout_positions(allocator: std.mem.Allocator, positions: []f64, n_type: NodeNeuronType, layout: EvolvableSubstrateLayout) !void {
    var count = positions.len / 2;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        var pos = try layout.node_position(allocator, i, n_type);
        defer pos.deinit();
        try std.testing.expect(pos.x == positions[i * 2] and pos.y == positions[i * 2 + 1]);
    }
}

test "MappedEvolvableSubstrateLayout node position" {
    const allocator = std.testing.allocator;
    const input_count: usize = 4;
    const output_count: usize = 2;
    var layout = EvolvableSubstrateLayout.init(try MappedEvolvableSubstrateLayout.init(allocator, input_count, output_count));
    defer layout.deinit();

    // check INPUT
    var d1 = [_]f64{ -0.75, -1, -0.25, -1, 0.25, -1, 0.75, -1 };
    try check_neuron_layout_positions(allocator, &d1, .InputNeuron, layout);
    try std.testing.expectError(error.NeuronIndexOutOfRange, layout.node_position(allocator, input_count, .InputNeuron));

    // check OUTPUT
    var d2 = [_]f64{ -0.5, 1, 0.5, 1 };
    try check_neuron_layout_positions(allocator, &d2, .OutputNeuron, layout);
    try std.testing.expectError(error.NeuronIndexOutOfRange, layout.node_position(allocator, output_count, .OutputNeuron));
}

test "MappedEvolvableSubstrateLayout add hidden node" {
    const allocator = std.testing.allocator;
    const input_count: usize = 4;
    const output_count: usize = 2;

    var layout = EvolvableSubstrateLayout.init(try MappedEvolvableSubstrateLayout.init(allocator, input_count, output_count));
    defer layout.deinit();

    var index: usize = 0;
    var x: f64 = -0.7;
    while (x < 0.8) : (x += 0.1) {
        var point = try PointF.init(allocator, x, 0);
        var h_index = try layout.add_hidden_node(point);
        try std.testing.expect(index == h_index);
        index += 1;
    }
    try std.testing.expect(index == layout.hidden_count());

    // test get hidden
    var delta: f64 = 0.0000000001;
    var i: usize = 0;
    while (i < index) : (i += 1) {
        x = -0.7 + @as(f64, @floatFromInt(i)) * 0.1;
        var h_point = try layout.node_position(allocator, i, .HiddenNeuron);
        try std.testing.expectApproxEqAbs(x, h_point.x, delta);
        try std.testing.expectApproxEqAbs(@as(f64, 0), h_point.y, delta);
    }

    // test index of
    index = 0;
    x = -0.7;
    while (x < 0.8) : (x += 0.1) {
        var point = try PointF.init(allocator, x, 0);
        defer point.deinit();
        var h_index = try layout.index_of_hidden(point);
        try std.testing.expect(h_index == index);
        index += 1;
    }
}
