const std = @import("std");

pub fn IdxMapType(comptime IdType: type) type {
    return if (IdType == []const u8) std.StringHashMap(usize) else std.AutoHashMap(IdType, usize);
}

pub inline fn getInfValue(comptime WeightType: type) WeightType {
    comptime {
        const inf_u32: u32 = @bitCast(std.math.inf(f32));
        const inf_u64: u64 = @bitCast(std.math.inf(f64));
        return switch (WeightType) {
            f16, f32, f64 => |T| std.math.inf(T),
            i32 => @intCast(inf_u32),
            u32 => inf_u32,
            i64 => @intCast(inf_u64),
            u64 => inf_u64,
            else => |T| @compileError("Unsupported type for WeightType: " ++ @typeName(T)),
        };
    }
}

test getInfValue {
    comptime {
        const types = [_]type{ f16, f32, f64 } ++ .{ i32, i64 } ++ .{ u32, u64 };
        for (types) |T| {
            _ = getInfValue(T); // just test for compilation errors
        }
    }
}

pub fn FifoQueue(comptime IdType: type, comptime Node: type) type {
    const ContainsMap = if (IdType == []const u8) std.StringHashMap(bool) else std.AutoHashMap(IdType, bool);

    return struct {
        queue: std.fifo.LinearFifo(*Node, .Dynamic),
        map: ContainsMap,
        allocator: std.mem.Allocator,

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator) !*Self {
            var self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .queue = std.fifo.LinearFifo(*Node, .Dynamic).init(allocator),
                .map = if (IdType == []const u8) std.StringHashMap(bool).init(allocator) else std.AutoHashMap(IdType, bool).init(allocator),
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.map.deinit();
            self.queue.deinit();
            self.allocator.destroy(self);
        }

        pub fn count(self: *Self) usize {
            return self.queue.count;
        }

        pub fn contains(self: *Self, node: *Node) bool {
            return self.map.contains(node.id);
        }

        pub fn push(self: *Self, node: *Node) !void {
            if (self.map.contains(node.id)) {
                return;
            }
            try self.queue.writeItem(node);
            _ = try self.map.put(node.id, true);
        }

        pub fn is_empty(self: *Self) bool {
            return self.queue.count == 0;
        }

        pub fn pop(self: *Self) ?*Node {
            var node: *Node = self.queue.readItem().?;
            _ = self.map.remove(node.id);
            return node;
        }
    };
}
