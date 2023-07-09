const std = @import("std");

pub const TransientAllocator = struct {
    count: u32 = 0,
    backing_alloc: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) TransientAllocator {
        return .{
            .backing_alloc = std.heap.ArenaAllocator.init(backing_allocator),
        };
    }

    pub fn deinit(self: *TransientAllocator) void {
        self.backing_alloc.deinit();
    }

    pub fn allocator(self: *TransientAllocator) std.mem.Allocator {
        return self.backing_alloc.allocator();
    }
};
