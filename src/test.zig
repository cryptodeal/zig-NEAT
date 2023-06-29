const std = @import("std");
const ThreadPool = std.Thread.Pool;
const WaitGroup = std.Thread.WaitGroup;

pub const WorkerCtx = struct {
    mu: std.Thread.Mutex = .{},
    items: []?*WorkerRes,
    allocator: std.heap.ThreadSafeAllocator,

    pub fn init(allocator: std.mem.Allocator, count: usize) !*WorkerCtx {
        var self: *WorkerCtx = try allocator.create(WorkerCtx);
        var thread_safe_allocator = std.heap.ThreadSafeAllocator{ .child_allocator = allocator };
        self.* = .{
            .items = try allocator.alloc(?*WorkerRes, count),
            .allocator = thread_safe_allocator,
        };
        return self;
    }

    pub fn deinit(self: *WorkerCtx) void {
        self.allocator.child_allocator.free(self.items);
        self.allocator.child_allocator.destroy(self);
    }
};

pub const WorkerRes = struct {
    worker_num: usize,
};

fn workerFn(i: usize, wg: *WaitGroup, ctx: *WorkerCtx) void {
    defer wg.finish();
    var res: ?*WorkerRes = ctx.allocator.allocator().create(WorkerRes) catch null;
    if (res != null) {
        res.?.* = .{ .worker_num = i };
    }
    ctx.mu.lock();
    defer ctx.mu.unlock();
    ctx.items[i] = res;
    std.debug.print("Hello from worker# {d}\n", .{i});
}

pub fn test_concurrency(allocator: std.mem.Allocator) !void {
    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();

    var wait_group: WaitGroup = .{};
    var ctx = try WorkerCtx.init(allocator, 10);
    defer ctx.deinit();
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        wait_group.start();
        try pool.spawn(workerFn, .{ i, &wait_group, ctx });
    }
    wait_group.wait();

    for (ctx.items) |item| {
        std.debug.print("item: {any}\n", .{item});
        if (item != null) {
            allocator.destroy(item.?);
        }
    }
}

test "concurrency" {
    try test_concurrency(std.testing.allocator);
}
