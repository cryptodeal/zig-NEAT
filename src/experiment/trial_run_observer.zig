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
    trial_run_started: *const fn (ctx: *anyopaque, trial: *Trial) void,

    /// Invoked to notify that the trial run just finished.
    /// Invoked after all epochs evaluated or successful solver found.
    trial_run_finished: *const fn (ctx: *anyopaque, trial: *Trial) void,

    /// Invoked to notify that evaluation of specific epoch completed.
    epoch_evaluated: *const fn (ctx: *anyopaque, trial: *Trial, epoch: *Generation) void,
};

// define interface methods wrapping vtable calls
pub fn trial_run_started(self: TrialRunObserver, trial: *Trial) void {
    self.vtable.trial_run_started(self.ptr, trial);
}

pub fn trial_run_finished(self: TrialRunObserver, trial: *Trial) void {
    self.vtable.trial_run_finished(self.ptr, trial);
}

pub fn epoch_evaluated(self: TrialRunObserver, trial: *Trial, epoch: *Generation) void {
    self.vtable.epoch_evaluated(self.ptr, trial, epoch);
}

pub fn init(observer: anytype) TrialRunObserver {
    const Ptr = @TypeOf(observer);
    const PtrInfo = @typeInfo(Ptr);
    assert(PtrInfo == .Pointer); // Must be a pointer
    assert(PtrInfo.Pointer.size == .One); // Must be a single-item pointer
    assert(@typeInfo(PtrInfo.Pointer.child) == .Struct); // Must point to a struct
    const impl = struct {
        fn trial_run_started(ctx: *anyopaque, trial: *Trial) void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            self.trial_run_started(trial);
        }

        fn trial_run_finished(ctx: *anyopaque, trial: *Trial) void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            self.trial_run_finished(trial);
        }

        fn epoch_evaluated(ctx: *anyopaque, trial: *Trial, epoch: *Generation) void {
            const self: Ptr = @ptrCast(@alignCast(ctx));
            self.epoch_evaluated(trial, epoch);
        }
    };
    return .{
        .ptr = observer,
        .vtable = &.{
            .trial_run_started = impl.trial_run_started,
            .trial_run_finished = impl.trial_run_finished,
            .epoch_evaluated = impl.epoch_evaluated,
        },
    };
}
