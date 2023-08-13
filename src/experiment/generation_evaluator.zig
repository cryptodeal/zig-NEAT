const std = @import("std");
const Trial = @import("trial.zig").Trial;
const Generation = @import("generation.zig").Generation;
const Options = @import("../opts.zig").Options;
const Population = @import("../genetics/population.zig").Population;
const assert = std.debug.assert;

const GenerationEvaluator = @This();

// The type erased pointer to the GenerationEvaluator implementation
ptr: *anyopaque,
vtable: *const VTable,

pub const VTable = struct {
    /// Invoked to evaluate one generation of population of organisms
    /// within given execution context.
    generationEvaluate: *const fn (ctx: *anyopaque, opts: *Options, pop: *Population, epoch: *Generation) anyerror!void,
};

// define interface methods wrapping vtable calls
pub fn generationEvaluate(self: GenerationEvaluator, opts: *Options, pop: *Population, epoch: *Generation) !void {
    try self.vtable.generationEvaluate(self.ptr, opts, pop, epoch);
}

pub fn init(generation_evaluator: anytype) GenerationEvaluator {
    const Ptr = @TypeOf(generation_evaluator);
    const PtrInfo = @typeInfo(Ptr);
    assert(PtrInfo == .Pointer); // Must be a pointer
    assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
    assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
    const impl = struct {
        fn generationEvaluate(ctx: *anyopaque, opts: *Options, pop: *Population, epoch: *Generation) !void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            try self.generationEvaluate(opts, pop, epoch);
        }
    };
    return .{
        .ptr = generation_evaluator,
        .vtable = &.{
            .generationEvaluate = impl.generationEvaluate,
        },
    };
}
