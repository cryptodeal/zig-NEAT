const std = @import("std");
const utils = @import("utils.zig");

const getInfValue = utils.getInfValue;
const IdxMapType = utils.IdxMapType;

pub fn Between(comptime WeightType: type, comptime NodeType: type) type {
    return struct {
        path: ?std.ArrayList(*NodeType) = null,
        weight: WeightType,
        unique: bool,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, path: ?std.ArrayList(*NodeType), wt: WeightType, unique: bool) !*Self {
            var self: *Self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .path = path,
                .weight = wt,
                .unique = unique,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.path != null) {
                self.path.?.deinit();
            }
            self.allocator.destroy(self);
        }
    };
}

pub fn AllBetween(comptime WeightType: type, comptime NodeType: type) type {
    const Inf_Val = comptime getInfValue(WeightType);

    return struct {
        paths: ?std.ArrayList(std.ArrayList(*NodeType)) = null,
        weight: WeightType = Inf_Val,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) !*Self {
            var self: *Self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.paths != null) {
                for (self.paths.?.items) |p| {
                    p.deinit();
                }
                self.paths.?.deinit();
            }
            self.allocator.destroy(self);
        }

        pub fn addPath(self: *Self, path: *std.ArrayList(*NodeType)) !void {
            if (self.paths == null) {
                self.paths = std.ArrayList(std.ArrayList(*NodeType)).init(self.allocator);
            }
            try self.paths.?.append(path.*);
        }
    };
}

pub fn PathTo(comptime WeightType: type, comptime NodeType: type) type {
    return struct {
        weight: WeightType,
        path: std.ArrayList(*NodeType),
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, path: *std.ArrayList(*NodeType), weight: WeightType) !*Self {
            var self: *Self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .path = path.*,
                .weight = weight,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.path.deinit();
            self.allocator.destroy(self);
        }
    };
}

pub fn Shortest(comptime WeightType: type, comptime IdType: type, comptime NodeType: type) type {
    return struct {
        const DistType = if (IdType == []const u8) std.StringHashMap(WeightType) else std.AutoHashMap(IdType, WeightType);
        const PrevType = if (IdType == []const u8) std.StringHashMap(*NodeType) else std.AutoHashMap(IdType, *NodeType);
        const VerticesMap = if (IdType == []const u8) ?std.StringHashMap(*NodeType) else ?std.AutoHashMap(IdType, *NodeType);

        src: *NodeType,
        nodes: VerticesMap,
        dist: DistType,
        prev: PrevType,
        has_negative_cycle: bool = false,
        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, src: *NodeType, dist: DistType, prev: PrevType, vertices: VerticesMap) !*Self {
            var self: *Self = try allocator.create(Self);
            self.* = .{
                .src = src,
                .allocator = allocator,
                .nodes = vertices,
                .dist = dist,
                .prev = prev,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.dist.deinit();
            self.prev.deinit();
            self.allocator.destroy(self);
        }

        pub fn from(self: *Self) *NodeType {
            return self.src;
        }

        pub fn weightTo(self: *Self, dst: IdType) WeightType {
            return self.dist.get(dst).?;
        }

        pub fn pathTo(self: *Self, dst: IdType) !*PathTo(WeightType, NodeType) {
            var path = std.ArrayList(*NodeType).init(self.allocator);
            if (!self.nodes.?.contains(dst)) {
                std.debug.print("Vertice not found in graph", .{});
                return error.VertexNotFound;
            }
            var dest = self.nodes.?.get(dst).?;
            var from_n = self.nodes.?.get(self.src.id).?;
            try path.append(self.nodes.?.get(dst).?);
            // var weight: WeightType = 99999999;
            var n = self.nodes.?.count();
            while (dest.id != from_n.id) {
                dest = self.prev.get(dest.id).?;
                try path.append(dest);
                if (n < 0) {
                    std.debug.print("path: unexpected negative cycle", .{});
                    return error.NegativeWeightCycle;
                }
                n -= 1;
            }
            std.mem.reverse(*NodeType, path.items);
            return PathTo(WeightType, NodeType).init(self.allocator, &path, self.dist.get(dst).?);
        }
    };
}

pub fn ShortestAlts(comptime IdType: type, comptime WeightType: type, comptime NodeType: type) type {
    const Inf_Val = comptime getInfValue(WeightType);

    return struct {
        src: *NodeType,
        nodes: []*NodeType,
        index_of: IdxMapType(IdType),
        N: usize,
        dist: []WeightType,
        next: []?std.ArrayList(usize),
        has_negative_cycle: bool = false,

        const Self = @This();

        pub fn init(allocator: std.Mem.Allocator, src: *NodeType, N: usize) !*Self {
            var alt = try allocator.create(Self);
            var next = try allocator.alloc(?std.ArrayList(usize), N);
            var dist = try allocator.alloc(WeightType, N);
            var i: usize = 0;
            while (i < N) : (i += 1) {
                next[i] = null;
                dist[i] = Inf_Val;
            }
            alt.* = .{
                .src = src,
                .nodes = try allocator.alloc(*NodeType, N),
                .next = next,
                .dist = dist,
                .N = N,
                .index_of = if (IdType == []const u8) std.StringHashMap(usize).init(allocator) else std.AutoHashMap(IdType, usize).init(allocator),
            };
            return alt;
        }

        pub fn deinit(self: *Self) void {
            self.index_of.deinit();
            self.allocator.free(self.nodes);
            self.allocator.free(self.dist);
            for (self.next) |n| {
                if (n != null) {
                    n.?.deinit();
                }
            }
            self.allocator.free(self.next);
            self.allocator.destroy(self);
        }
    };
}

pub fn AllShortest(comptime IdType: type, comptime WeightType: type, comptime NodeType: type) type {
    const Inf_Val = comptime getInfValue(WeightType);

    return struct {
        allocator: std.mem.Allocator,
        N: usize,
        nodes: []*NodeType,
        dist: [][]WeightType,
        index_of: IdxMapType(IdType),
        next: []?std.ArrayList(usize),
        forward: bool,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, N: usize, forward: bool) !*Self {
            var self: *Self = try allocator.create(Self);
            var dist: [][]WeightType = try allocator.alloc([]WeightType, N);
            for (dist, 0..) |_, i| {
                var arr = try allocator.alloc(WeightType, N);
                for (arr) |*x| x.* = Inf_Val;
                dist[i] = arr;
            }

            var next = try allocator.alloc(?std.ArrayList(usize), N * N);
            for (next) |*x| x.* = null;

            self.* = .{
                .allocator = allocator,
                .nodes = try allocator.alloc(*NodeType, N),
                .dist = dist,
                .N = N,
                .index_of = if (IdType == []const u8) std.StringHashMap(usize).init(allocator) else std.AutoHashMap(IdType, usize).init(allocator),
                .next = next,
                .forward = forward,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (self.dist) |arr| {
                self.allocator.free(arr);
            }
            self.allocator.free(self.dist);
            for (self.next) |arr| {
                if (arr != null) {
                    arr.?.deinit();
                }
            }
            self.allocator.free(self.next);
            self.allocator.free(self.nodes);
            self.index_of.deinit();
            self.allocator.destroy(self);
        }

        pub fn at(self: *Self, src: usize, dst: usize) !std.ArrayList(usize) {
            var og = self.next[src + dst * self.nodes.len];
            if (og == null) {
                return std.ArrayList(usize).init(self.allocator);
            }
            return og.?.clone();
        }

        pub fn set(self: *Self, src: usize, dst: usize, wt: WeightType, mid: std.ArrayList(usize)) void {
            self.dist[src][dst] = wt;
            if (self.next[src + dst * self.nodes.len] != null) {
                self.next[src + dst * self.nodes.len].?.deinit();
            }
            self.next[src + dst * self.nodes.len] = mid;
        }

        pub fn add(self: *Self, src: usize, dst: usize, mid: std.ArrayList(usize)) !void {
            defer mid.deinit();
            loop: for (mid.items) |k| {
                for (self.next[src + dst * self.nodes.len].?.items) |v| {
                    if (k == v) {
                        continue :loop;
                    }
                }
                try self.next[src + dst * self.nodes.len].?.append(k);
            }
        }

        pub fn weight(self: *Self, src: IdType, dst: IdType) WeightType {
            var from = self.index_of.get(src);
            var to = self.index_of.get(dst);
            if (from == null or to == null) {
                return Inf_Val;
            }
            return self.dist[from.?][to.?];
        }

        pub fn between(self: *Self, src: IdType, dst: IdType) !*Between(WeightType, NodeType) {
            var prng = std.rand.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                try std.os.getrandom(std.mem.asBytes(&seed));
                break :blk seed;
            });
            const rand = prng.random();
            var from = self.index_of.get(src);
            var to = self.index_of.get(dst);
            var test_path = try self.at(from.?, to.?);
            defer test_path.deinit();
            if (from == null or to == null or test_path.items.len == 0) {
                if (src == dst) {
                    var path = std.ArrayList(*NodeType).init(self.allocator);
                    try path.append(self.nodes[from.?]);
                    return Between(WeightType, NodeType).init(self.allocator, path, 0, true);
                }
                return Between(WeightType, NodeType).init(self.allocator, null, Inf_Val, false);
            }

            var wt: WeightType = self.dist[from.?][to.?];
            if (wt == -Inf_Val) {
                return Between(WeightType, NodeType).init(self.allocator, null, wt, false);
            }

            var seen = try self.allocator.alloc(i64, self.nodes.len);
            defer self.allocator.free(seen);
            for (seen, 0..) |_, i| {
                seen[i] = -1;
            }

            var n: *NodeType = undefined;
            if (self.forward) {
                n = self.nodes[from.?];
                seen[from.?] = 0;
            } else {
                n = self.nodes[to.?];
                seen[to.?] = 0;
            }

            var path = std.ArrayList(*NodeType).init(self.allocator);
            try path.append(n);
            var unique = true;

            var next: usize = undefined;
            while (from.? != to.?) {
                var c: std.ArrayList(usize) = try self.at(from.?, to.?);
                defer c.deinit();
                if (c.items.len != 1) {
                    unique = false;
                    next = c.items[rand.uintLessThan(usize, c.items.len)];
                } else {
                    next = c.items[0];
                }
                if (seen[next] >= 0) {
                    try path.resize(@as(usize, @intCast(seen[next])));
                    // TODO: verify next line is not necessary
                    // path.items = path.items[0..seen[next]];
                }
                seen[next] = @as(i64, @intCast(path.items.len));
                try path.append(self.nodes[next]);
                if (self.forward) {
                    from = next;
                } else {
                    to = next;
                }
            }
            if (!self.forward) {
                std.mem.reverse(*NodeType, path.items);
            }
            return Between(WeightType, NodeType).init(self.allocator, path, wt, unique);
        }

        fn allBetweenCb(res: *AllBetween(WeightType, NodeType), path: *std.ArrayList(*NodeType)) !void {
            try res.addPath(path);
        }

        pub fn allBetween(self: *Self, src: IdType, dst: IdType) !*AllBetween(WeightType, NodeType) {
            var res = try AllBetween(WeightType, NodeType).init(self.allocator);
            var from = self.index_of.get(src);
            var to = self.index_of.get(dst);
            var test_path = try self.at(from.?, to.?);
            defer test_path.deinit();
            if (from == null or to == null or test_path.items.len == 0) {
                if (src == dst) {
                    var path = std.ArrayList(*NodeType).init(self.allocator);
                    try path.append(self.nodes[from.?]);
                    try res.addPath(&path);
                    res.weight = 0;
                    return res;
                }
                return res;
            }
            res.weight = self.dist[from.?][to.?];
            var n: *NodeType = undefined;
            if (self.forward) {
                n = self.nodes[from.?];
            } else {
                n = self.nodes[to.?];
            }
            var seen = try self.allocator.alloc(bool, self.N);
            defer self.allocator.free(seen);

            var path: std.ArrayList(*NodeType) = std.ArrayList(*NodeType).init(self.allocator);
            try path.append(n);

            try self.all_between(from.?, to.?, seen, &path, res, allBetweenCb);

            return res;
        }

        fn all_between(self: *Self, from: usize, to: usize, seen: []bool, path: ?*std.ArrayList(*NodeType), res: *AllBetween(WeightType, NodeType), comptime func: fn (*AllBetween(WeightType, NodeType), *std.ArrayList(*NodeType)) std.mem.Allocator.Error!void) !void {
            if (self.forward) {
                seen[from] = true;
            } else {
                seen[to] = true;
            }
            if (from == to) {
                if (path == null) {
                    return;
                }
                if (!self.forward) {
                    std.mem.reverse(*NodeType, path.?.*.items);
                }
                try func(res, path.?);
                if (!self.forward) {
                    std.mem.reverse(*NodeType, path.?.*.items);
                }
                return;
            }
            var first = true;
            var seen_work: ?[]bool = null;
            var p_at = try self.at(from, to);
            var used_path: std.ArrayList(*NodeType) = undefined;
            var src: usize = from;
            var dst: usize = to;
            defer p_at.deinit();
            for (p_at.items) |n| {
                if (seen[n]) {
                    continue;
                }
                if (first) {
                    used_path = try path.?.*.clone();
                    seen_work = try self.allocator.alloc(bool, self.N);
                    first = false;
                }
                if (self.forward) {
                    src = n;
                } else {
                    dst = n;
                }
                @memcpy(seen_work.?, seen);
                try used_path.append(self.nodes[n]);
                try self.all_between(src, dst, seen_work.?, &used_path, res, func);
            }
            path.?.*.deinit();
            if (seen_work != null) {
                self.allocator.free(seen_work.?);
            }
        }
    };
}
