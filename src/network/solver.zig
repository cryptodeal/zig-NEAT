const std = @import("std");
const assert = std.debug.assert;

pub const Solver = @This();

// The type erased pointer to the GenerationEvaluator implementation
ptr: *anyopaque,
vtable: *const VTable,

pub const VTable = struct {
    /// Propagates activation wave through all network nodes provided number of steps in forward direction.
    /// Normally the number of steps should be equal to the activation depth of the network.
    /// Returns true if activation wave passed from all inputs to the output nodes.
    forward_steps: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, steps: usize) anyerror!bool,

    /// Propagates activation wave through all network nodes provided number of steps by recursion from output nodes
    /// Returns true if activation wave passed from all inputs to the output nodes.
    recursive_steps: *const fn (ctx: *anyopaque) anyerror!bool,

    /// Attempts to relax network given amount of steps until giving up. The network considered relaxed when absolute
    /// value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
    /// If maxAllowedSignalDelta value is less than or equal to 0, the method will return true without checking for relaxation.
    relax: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, max_steps: usize, max_allowed_signal_delta: f64) anyerror!bool,

    /// Flushes network state by removing all current activations. Returns true if network flushed successfully or
    /// false in case of error.
    flush: *const fn (ctx: *anyopaque) anyerror!bool,

    /// Set sensors values to the input nodes of the network
    load_sensors: *const fn (ctx: *anyopaque, inputs: []f64) anyerror!void,

    /// Read output values from the output nodes of the network
    read_outputs: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]f64,

    /// Returns the total number of neural units in the network
    node_count: *const fn (ctx: *anyopaque) usize,

    /// Returns the total number of links between nodes in the network
    link_count: *const fn (ctx: *anyopaque) usize,

    /// Frees all memory from Solver implementation
    deinit: *const fn (ctx: *anyopaque) void,
};

// define interface methods wrapping vtable calls
pub fn forward_steps(self: Solver, allocator: std.mem.Allocator, steps: usize) !bool {
    return self.vtable.forward_steps(self.ptr, allocator, steps);
}

pub fn recursive_steps(self: Solver) !bool {
    return self.vtable.recursive_steps(self.ptr);
}

pub fn relax(self: Solver, allocator: std.mem.Allocator, max_steps: usize, max_allowed_signal_delta: f64) !bool {
    return self.vtable.relax(self.ptr, allocator, max_steps, max_allowed_signal_delta);
}

pub fn flush(self: Solver) !bool {
    return self.vtable.flush(self.ptr);
}

pub fn load_sensors(self: Solver, inputs: []f64) !void {
    return self.vtable.load_sensors(self.ptr, inputs);
}

pub fn read_outputs(self: Solver, allocator: std.mem.Allocator) ![]f64 {
    return self.vtable.read_outputs(self.ptr, allocator);
}

pub fn node_count(self: Solver) usize {
    return self.vtable.node_count(self.ptr);
}

pub fn link_count(self: Solver) usize {
    return self.vtable.link_count(self.ptr);
}

pub fn deinit(self: Solver) void {
    return self.vtable.deinit(self.ptr);
}

pub fn init(network_solver: anytype) Solver {
    const Ptr = @TypeOf(network_solver);
    const PtrInfo = @typeInfo(Ptr);
    assert(PtrInfo == .Pointer); // Must be a pointer
    assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
    assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
    const impl = struct {
        fn forward_steps(ctx: *anyopaque, allocator: std.mem.Allocator, steps: usize) !bool {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.forward_steps(allocator, steps);
        }

        fn recursive_steps(ctx: *anyopaque) !bool {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.recursive_steps();
        }

        fn relax(ctx: *anyopaque, allocator: std.mem.Allocator, max_steps: usize, max_allowed_signal_delta: f64) !bool {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.relax(allocator, max_steps, max_allowed_signal_delta);
        }

        fn flush(ctx: *anyopaque) !bool {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.flush();
        }

        fn load_sensors(ctx: *anyopaque, inputs: []f64) !void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            try self.load_sensors(inputs);
        }

        fn read_outputs(ctx: *anyopaque, allocator: std.mem.Allocator) ![]f64 {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.read_outputs(allocator);
        }

        fn node_count(ctx: *anyopaque) usize {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.node_count();
        }

        fn link_count(ctx: *anyopaque) usize {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.link_count();
        }

        fn deinit(ctx: *anyopaque) void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            self.deinit();
        }
    };
    return .{
        .ptr = network_solver,
        .vtable = &.{
            .forward_steps = impl.forward_steps,
            .recursive_steps = impl.recursive_steps,
            .relax = impl.relax,
            .flush = impl.flush,
            .load_sensors = impl.load_sensors,
            .read_outputs = impl.read_outputs,
            .node_count = impl.node_count,
            .link_count = impl.link_count,
            .deinit = impl.deinit,
        },
    };
}
