const std = @import("std");

pub const CancelCtx = struct {
    allocator: std.mem.Allocator,
    mu: std.Thread.Mutex = .{},
    context: Context,
    done: std.Thread.Cond = .{},
};

pub const EmptyCtx = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*EmptyCtx {
        var self = try allocator.create(EmptyCtx);
        self.* = .{
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *EmptyCtx) void {
        self.allocator.destroy(self);
    }
};

pub const Context = union(enum) {
    empty_ctx: *EmptyCtx,
    cancel_ctx: *CancelCtx,
};

pub fn ValueContext(comptime ContextType: type, comptime KeyType: type, comptime ValueType: type) type {
    return struct {
        const Self = @This();

        context: ContextType,
        key: KeyType,
        value: ValueType,

        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, context: ContextType, key: KeyType, val: ValueType) !*Self {
            var self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .context = context,
                .key = key,
                .value = val,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.destroy(self);
        }
    };
}
