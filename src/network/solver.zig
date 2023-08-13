//! The standard Solver interface.

const std = @import("std");
const assert = std.debug.assert;

pub const Solver = @This();

// The type erased pointer to the Solver implementation
ptr: *anyopaque,
vtable: *const VTable,

pub const VTable = struct {
    /// Propagates activation wave through all Network nodes by the provided number of steps in forward direction.
    /// Normally the number of steps should be equal to the activation depth of the Network.
    /// Returns true if activation wave passed from all inputs to the output nodes.
    forwardSteps: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, steps: usize) anyerror!bool,

    /// Propagates activation wave through all Network nodes by the provided number of steps by recursion from output nodes
    /// Returns true if activation wave passed from all inputs to the output nodes.
    recursiveSteps: *const fn (ctx: *anyopaque) anyerror!bool,

    /// Attempts to relax Network given number of steps until giving up. The Network is considered relaxed when absolute
    /// value of the change at any given point is less than `max_allowed_signal_delta` during activation waves propagation.
    /// If `max_allowed_signal_delta` value is less than or equal to 0, the method will return true without checking for relaxation.
    relax: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, max_steps: usize, max_allowed_signal_delta: f64) anyerror!bool,

    /// Flushes Network state by removing all current activations. Returns true if Network flushed successfully or
    /// error.
    flush: *const fn (ctx: *anyopaque) anyerror!bool,

    /// Set sensors values to the input nodes of the Network.
    loadSensors: *const fn (ctx: *anyopaque, inputs: []f64) anyerror!void,

    /// Read output values from the output nodes of the Network.
    readOutputs: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror![]f64,

    /// Returns the total number of neural units in the Network.
    nodeCount: *const fn (ctx: *anyopaque) usize,

    /// Returns the total number of links between nodes in the Network.
    linkCount: *const fn (ctx: *anyopaque) usize,

    /// Frees all associated memory from underlying implementation.
    deinit: *const fn (ctx: *anyopaque) void,
};

// define interface methods wrapping vtable calls

/// Propagates activation wave through all Network nodes by the provided number of steps in forward direction.
/// Normally the number of steps should be equal to the activation depth of the Network.
/// Returns true if activation wave passed from all inputs to the output nodes.
pub fn forwardSteps(self: Solver, allocator: std.mem.Allocator, steps: usize) !bool {
    return self.vtable.forwardSteps(self.ptr, allocator, steps);
}

/// Propagates activation wave through all Network nodes by the provided number of steps by recursion from output nodes
/// Returns true if activation wave passed from all inputs to the output nodes.
pub fn recursiveSteps(self: Solver) !bool {
    return self.vtable.recursiveSteps(self.ptr);
}

/// Attempts to relax Network a given number of steps until giving up. The Network is considered relaxed when absolute
/// value of the change at any given point is less than `max_allowed_signal_delta` during activation waves propagation.
/// If `max_allowed_signal_delta` value is less than or equal to 0, the method will return true without checking for relaxation.
pub fn relax(self: Solver, allocator: std.mem.Allocator, max_steps: usize, max_allowed_signal_delta: f64) !bool {
    return self.vtable.relax(self.ptr, allocator, max_steps, max_allowed_signal_delta);
}

/// Flushes Network state by removing all current activations. Returns true if Network flushed successfully or
/// error.
pub fn flush(self: Solver) !bool {
    return self.vtable.flush(self.ptr);
}

/// Set sensors values to the input nodes of the Network.
pub fn loadSensors(self: Solver, inputs: []f64) !void {
    return self.vtable.loadSensors(self.ptr, inputs);
}

/// Read output values from the output nodes of the Network.
pub fn readOutputs(self: Solver, allocator: std.mem.Allocator) ![]f64 {
    return self.vtable.readOutputs(self.ptr, allocator);
}

/// Returns the total number of neural units in the Network.
pub fn nodeCount(self: Solver) usize {
    return self.vtable.nodeCount(self.ptr);
}

/// Returns the total number of links between nodes in the Network.
pub fn linkCount(self: Solver) usize {
    return self.vtable.linkCount(self.ptr);
}

/// Frees all associated memory from underlying implementation.
pub fn deinit(self: Solver) void {
    return self.vtable.deinit(self.ptr);
}

/// Initializes a new Solver from the provided pointer to implementation.
pub fn init(network_solver: anytype) Solver {
    const Ptr = @TypeOf(network_solver);
    const PtrInfo = @typeInfo(Ptr);
    assert(PtrInfo == .Pointer); // Must be a pointer
    assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
    assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
    const impl = struct {
        fn forwardSteps(ctx: *anyopaque, allocator: std.mem.Allocator, steps: usize) !bool {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.forwardSteps(allocator, steps);
        }

        fn recursiveSteps(ctx: *anyopaque) !bool {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.recursiveSteps();
        }

        fn relax(ctx: *anyopaque, allocator: std.mem.Allocator, max_steps: usize, max_allowed_signal_delta: f64) !bool {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.relax(allocator, max_steps, max_allowed_signal_delta);
        }

        fn flush(ctx: *anyopaque) !bool {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.flush();
        }

        fn loadSensors(ctx: *anyopaque, inputs: []f64) !void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            try self.loadSensors(inputs);
        }

        fn readOutputs(ctx: *anyopaque, allocator: std.mem.Allocator) ![]f64 {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.readOutputs(allocator);
        }

        fn nodeCount(ctx: *anyopaque) usize {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.nodeCount();
        }

        fn linkCount(ctx: *anyopaque) usize {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            return self.linkCount();
        }

        fn deinit(ctx: *anyopaque) void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            self.deinit();
        }
    };

    return .{
        .ptr = network_solver,
        .vtable = &.{
            .forwardSteps = impl.forwardSteps,
            .recursiveSteps = impl.recursiveSteps,
            .relax = impl.relax,
            .flush = impl.flush,
            .loadSensors = impl.loadSensors,
            .readOutputs = impl.readOutputs,
            .nodeCount = impl.nodeCount,
            .linkCount = impl.linkCount,
            .deinit = impl.deinit,
        },
    };
}
