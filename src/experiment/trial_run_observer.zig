//! The statndard TrialRunObserver interface.

const std = @import("std");
const Trial = @import("trial.zig").Trial;
const Generation = @import("generation.zig").Generation;
const assert = std.debug.assert;

const TrialRunObserver = @This();

// The type erased pointer to the TrialRunObserver implementation
ptr: *anyopaque,
vtable: *const VTable,

pub const VTable = struct {
    /// Invoked to notify that new trial run just started.
    /// Invoked before any epoch evaluation in that trial run.
    trialRunStarted: *const fn (ctx: *anyopaque, trial: *Trial) void,

    /// Invoked to notify that the trial run just finished.
    /// Invoked after all epochs evaluated or successful solver found.
    trialRunFinished: *const fn (ctx: *anyopaque, trial: *Trial) void,

    /// Invoked to notify that evaluation of specific epoch completed.
    epochEvaluated: *const fn (ctx: *anyopaque, trial: *Trial, epoch: *Generation) void,
};

/// Invoked to notify that new trial run just started.
/// Invoked before any epoch evaluation in that trial run.
pub fn trialRunStarted(self: TrialRunObserver, trial: *Trial) void {
    self.vtable.trialRunStarted(self.ptr, trial);
}

/// Invoked to notify that the trial run just finished.
/// Invoked after all epochs evaluated or successful solver found.
pub fn trialRunFinished(self: TrialRunObserver, trial: *Trial) void {
    self.vtable.trialRunFinished(self.ptr, trial);
}

/// Invoked to notify that evaluation of specific epoch completed.
pub fn epochEvaluated(self: TrialRunObserver, trial: *Trial, epoch: *Generation) void {
    self.vtable.epochEvaluated(self.ptr, trial, epoch);
}

/// Initializes a new TrialRunObserver from the provided pointer to implementation.
pub fn init(observer: anytype) TrialRunObserver {
    const Ptr = @TypeOf(observer);
    const PtrInfo = @typeInfo(Ptr);
    assert(PtrInfo == .Pointer); // Must be a pointer
    assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
    assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
    const impl = struct {
        fn trialRunStarted(ctx: *anyopaque, trial: *Trial) void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            self.trialRunStarted(trial);
        }

        fn trialRunFinished(ctx: *anyopaque, trial: *Trial) void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            self.trialRunFinished(trial);
        }

        fn epochEvaluated(ctx: *anyopaque, trial: *Trial, epoch: *Generation) void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            self.epochEvaluated(trial, epoch);
        }
    };
    return .{
        .ptr = observer,
        .vtable = &.{
            .trialRunStarted = impl.trialRunStarted,
            .trialRunFinished = impl.trialRunFinished,
            .epochEvaluated = impl.epochEvaluated,
        },
    };
}
