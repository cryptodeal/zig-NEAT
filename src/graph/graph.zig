const std = @import("std");
const utils = @import("utils.zig");

const getInfValue = utils.getInfValue;
const FifoQueue = utils.FifoQueue;

pub const GraphError = error{
    VertexNotFound,
    EdgeNotFound,
    NegativeWeightCycle,
    NegativePathWeight,
};

pub fn Graph(comptime IdType: type, comptime DType: type, comptime WeightType: type) type {
    const Inf_Val = comptime getInfValue(WeightType);
    const IdxMapType = if (IdType == []const u8) std.StringHashMap(usize) else std.AutoHashMap(IdType, usize);

    return struct {
        const VerticesMap = if (IdType == []const u8) ?std.StringHashMap(*Node) else ?std.AutoHashMap(IdType, *Node);

        N: usize,
        connected: usize,
        root: ?*Node,
        vertices: VerticesMap,
        graph: ?std.AutoHashMap(*Node, std.ArrayList(*Edge)),
        allocator: std.mem.Allocator,

        const Self = @This();

        pub const Node = struct {
            id: IdType,
            data: DType,

            pub fn init(n: IdType, d: DType) Node {
                return Node{ .id = n, .data = d };
            }
        };

        pub const Edge = struct {
            node: *Node,
            weight: WeightType,

            pub fn init(n1: *Node, w: WeightType) Edge {
                return Edge{ .node = n1, .weight = w };
            }

            pub fn clone(self: *Edge, allocator: std.mem.Allocator) !*Edge {
                var cloned: *Edge = try allocator.create(Edge);
                cloned.* = .{
                    .node = self.node,
                    .weight = self.weight,
                };
                return cloned;
            }
        };

        const AllShortest = struct {
            allocator: std.mem.Allocator,
            N: usize,
            nodes: []*Node,
            dist: [][]WeightType,
            index_of: IdxMapType,
            next: []?std.ArrayList(usize),
            forward: bool,

            pub fn init(allocator: std.mem.Allocator, N: usize, forward: bool) !*AllShortest {
                var self: *AllShortest = try allocator.create(AllShortest);
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
                    .nodes = try allocator.alloc(*Node, N),
                    .dist = dist,
                    .N = N,
                    .index_of = if (IdType == []const u8) std.StringHashMap(usize).init(allocator) else std.AutoHashMap(IdType, usize).init(allocator),
                    .next = next,
                    .forward = forward,
                };
                return self;
            }

            pub fn deinit(self: *AllShortest) void {
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

            pub fn at(self: *AllShortest, src: usize, dst: usize) !std.ArrayList(usize) {
                var og = self.next[src + dst * self.nodes.len];
                if (og == null) {
                    return std.ArrayList(usize).init(self.allocator);
                }
                return try og.?.clone();
            }

            pub fn set(self: *AllShortest, src: usize, dst: usize, wt: WeightType, mid: std.ArrayList(usize)) void {
                self.dist[src][dst] = wt;
                if (self.next[src + dst * self.nodes.len] != null) {
                    self.next[src + dst * self.nodes.len].?.deinit();
                }
                self.next[src + dst * self.nodes.len] = mid;
            }

            pub fn add(self: *AllShortest, src: usize, dst: usize, mid: std.ArrayList(usize)) !void {
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

            pub fn weight(self: *AllShortest, src: IdType, dst: IdType) WeightType {
                var from = self.index_of.get(src);
                var to = self.index_of.get(dst);
                if (from == null or to == null) {
                    return Inf_Val;
                }
                return self.dist[from.?][to.?];
            }

            pub fn between(self: *AllShortest, src: IdType, dst: IdType) !*Between {
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
                        var path = std.ArrayList(*Node).init(self.allocator);
                        try path.append(self.nodes[from.?]);
                        return try Between.init(self.allocator, path, 0, true);
                    }
                    return try Between.init(self.allocator, null, Inf_Val, false);
                }

                var wt: WeightType = self.dist[from.?][to.?];
                if (wt == -Inf_Val) {
                    return try Between.init(self.allocator, null, wt, false);
                }

                var seen = try self.allocator.alloc(i64, self.nodes.len);
                defer self.allocator.free(seen);
                for (seen, 0..) |_, i| {
                    seen[i] = -1;
                }

                var n: *Node = undefined;
                if (self.forward) {
                    n = self.nodes[from.?];
                    seen[from.?] = 0;
                } else {
                    n = self.nodes[to.?];
                    seen[to.?] = 0;
                }

                var path = std.ArrayList(*Node).init(self.allocator);
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
                    std.mem.reverse(*Node, path.items);
                }
                return try Between.init(self.allocator, path, wt, unique);
            }

            const AllBetween = struct {
                paths: ?std.ArrayList(std.ArrayList(*Node)) = null,
                weight: WeightType = Inf_Val,
                allocator: std.mem.Allocator,

                pub fn init(allocator: std.mem.Allocator) !*AllBetween {
                    var self: *AllBetween = try allocator.create(AllBetween);
                    self.* = .{
                        .allocator = allocator,
                    };
                    return self;
                }

                pub fn deinit(self: *AllBetween) void {
                    if (self.paths != null) {
                        for (self.paths.?.items) |p| {
                            p.deinit();
                        }
                        self.paths.?.deinit();
                    }
                    self.allocator.destroy(self);
                }

                pub fn add_path(self: *AllBetween, path: *std.ArrayList(*Node)) !void {
                    if (self.paths == null) {
                        self.paths = std.ArrayList(std.ArrayList(*Node)).init(self.allocator);
                    }
                    try self.paths.?.append(path.*);
                }
            };

            fn allBetweenCb(res: *AllBetween, path: *std.ArrayList(*Node)) !void {
                try res.add_path(path);
            }

            pub fn all_between(self: *AllShortest, src: IdType, dst: IdType) !*AllBetween {
                var res = try AllBetween.init(self.allocator);
                var from = self.index_of.get(src);
                var to = self.index_of.get(dst);
                var test_path = try self.at(from.?, to.?);
                defer test_path.deinit();
                if (from == null or to == null or test_path.items.len == 0) {
                    if (src == dst) {
                        var path = std.ArrayList(*Node).init(self.allocator);
                        try path.append(self.nodes[from.?]);
                        try res.add_path(&path);
                        res.weight = 0;
                        return res;
                    }
                    return res;
                }
                res.weight = self.dist[from.?][to.?];
                var n: *Node = undefined;
                if (self.forward) {
                    n = self.nodes[from.?];
                } else {
                    n = self.nodes[to.?];
                }
                var seen = try self.allocator.alloc(bool, self.N);
                defer self.allocator.free(seen);

                var path: std.ArrayList(*Node) = std.ArrayList(*Node).init(self.allocator);
                try path.append(n);

                try self.allBetween(from.?, to.?, seen, &path, res, allBetweenCb);

                return res;
            }

            fn allBetween(self: *AllShortest, from: usize, to: usize, seen: []bool, path: ?*std.ArrayList(*Node), res: *AllBetween, comptime func: fn (*AllBetween, *std.ArrayList(*Node)) std.mem.Allocator.Error!void) !void {
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
                        std.mem.reverse(*Node, path.?.*.items);
                    }
                    try func(res, path.?);
                    if (!self.forward) {
                        std.mem.reverse(*Node, path.?.*.items);
                    }
                    return;
                }
                var first = true;
                var seen_work: ?[]bool = null;
                var p_at = try self.at(from, to);
                var used_path: std.ArrayList(*Node) = undefined;
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
                    try self.allBetween(src, dst, seen_work.?, &used_path, res, func);
                }
                path.?.*.deinit();
                if (seen_work != null) {
                    self.allocator.free(seen_work.?);
                }
            }
        };

        const Shortest = struct {
            const DistType = if (IdType == []const u8) std.StringHashMap(WeightType) else std.AutoHashMap(IdType, WeightType);
            const PrevType = if (IdType == []const u8) std.StringHashMap(*Node) else std.AutoHashMap(IdType, *Node);

            src: *Node,
            nodes: VerticesMap,
            dist: DistType,
            prev: PrevType,
            has_negative_cycle: bool = false,
            allocator: std.mem.Allocator,

            pub fn init(allocator: std.mem.Allocator, src: *Node, dist: DistType, prev: PrevType, vertices: VerticesMap) !*Shortest {
                var self: *Shortest = try allocator.create(Shortest);
                self.* = .{
                    .src = src,
                    .allocator = allocator,
                    .nodes = vertices,
                    .dist = dist,
                    .prev = prev,
                };
                return self;
            }

            pub fn deinit(self: *Shortest) void {
                self.dist.deinit();
                self.prev.deinit();
                self.allocator.destroy(self);
            }

            pub fn from(self: *Shortest) *Node {
                return self.src;
            }

            pub fn weight_to(self: *Shortest, dst: IdType) WeightType {
                return self.dist.get(dst).?;
            }

            const PathTo = struct {
                weight: WeightType,
                path: std.ArrayList(*Node),
                allocator: std.mem.Allocator,

                pub fn init(allocator: std.mem.Allocator, path: *std.ArrayList(*Node), weight: WeightType) !*PathTo {
                    var self = try allocator.create(PathTo);
                    self.* = .{
                        .allocator = allocator,
                        .path = path.*,
                        .weight = weight,
                    };
                    return self;
                }

                pub fn deinit(self: *PathTo) void {
                    self.path.deinit();
                    self.allocator.destroy(self);
                }
            };

            pub fn path_to(self: *Shortest, dst: IdType) !*PathTo {
                var path = std.ArrayList(*Node).init(self.allocator);
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
                std.mem.reverse(*Node, path.items);
                return try PathTo.init(self.allocator, &path, self.dist.get(dst).?);
            }
        };

        const Between = struct {
            path: ?std.ArrayList(*Node) = null,
            weight: WeightType,
            unique: bool,
            allocator: std.mem.Allocator,

            pub fn init(allocator: std.mem.Allocator, path: ?std.ArrayList(*Node), wt: WeightType, unique: bool) !*Between {
                var self: *Between = try allocator.create(Between);
                self.* = .{
                    .allocator = allocator,
                    .path = path,
                    .weight = wt,
                    .unique = unique,
                };
                return self;
            }

            pub fn deinit(self: *Between) void {
                if (self.path != null) {
                    self.path.?.deinit();
                }
                self.allocator.destroy(self);
            }
        };

        const ShortestAlts = struct {
            src: *Node,
            nodes: []*Node,
            index_of: IdxMapType,
            N: usize,
            dist: []WeightType,
            next: []?std.ArrayList(usize),
            has_negative_cycle: bool = false,

            pub fn init(allocator: std.Mem.Allocator, src: *Node, N: usize) !*ShortestAlts {
                var alt = try allocator.create(ShortestAlts);
                var next = try allocator.alloc(?std.ArrayList(usize), N);
                var dist = try allocator.alloc(WeightType, N);
                var i: usize = 0;
                while (i < N) : (i += 1) {
                    next[i] = null;
                    dist[i] = Inf_Val;
                }
                alt.* = .{
                    .src = src,
                    .nodes = try allocator.alloc(*Node, N),
                    .next = next,
                    .dist = dist,
                    .N = N,
                    .index_of = if (IdType == []const u8) std.StringHashMap(usize).init(allocator) else std.AutoHashMap(IdType, usize).init(allocator),
                };
                return alt;
            }

            pub fn deinit(self: *ShortestAlts) void {
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

        pub fn init(alloc: std.mem.Allocator) Self {
            return Self{
                .N = 0,
                .connected = 0,
                .root = undefined,
                .vertices = undefined,
                .graph = undefined,
                .allocator = alloc,
            };
        }

        pub fn clone(self: *Self) !Self {
            var cloned_graph = try self.graph.?.clone();
            var graph_it = self.graph.?.iterator();
            while (graph_it.next()) |v| {
                var key: *Node = v.key_ptr.*;
                var edges_out: std.ArrayList(*Edge) = v.value_ptr.*;
                var cloned_edges_out = cloned_graph.get(key);
                for (edges_out.items, 0..) |e, i| {
                    cloned_edges_out.items[i] = try e.clone(self.allocator);
                }
            }
            var cloned = Self{
                .N = self.N,
                .connected = self.connected,
                .root = self.root,
                .vertices = try self.vertices.?.clone(),
                .graph = cloned_graph,
                .allocator = self.allocator,
            };
            return cloned;
        }

        pub fn deinit(self: *Self) void {
            var graph_it = self.graph.?.iterator();
            while (graph_it.next()) |entry| {
                // self.allocator.destroy(entry.key_ptr);
                for (entry.value_ptr.*.items) |v| {
                    self.allocator.destroy(v);
                }
                entry.value_ptr.*.deinit();
            }
            self.graph.?.deinit();

            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                self.allocator.destroy(entry.value_ptr.*);
            }
            self.vertices.?.deinit();

            self.N = 0;
        }

        pub fn add_vertex(self: *Self, n: IdType, d: DType) !void {
            if (self.N == 0) {
                var rt = try self.allocator.create(Node);
                errdefer self.allocator.destroy(rt);
                rt.* = Node.init(n, d);

                self.root = rt;
                self.vertices = if (IdType == []const u8) std.StringHashMap(*Node).init(self.allocator) else std.AutoHashMap(IdType, *Node).init(self.allocator);
                _ = try self.vertices.?.put(rt.id, rt);

                self.graph = std.AutoHashMap(*Node, std.ArrayList(*Edge)).init(self.allocator);
                _ = try self.graph.?.put(rt, std.ArrayList(*Edge).init(self.allocator));

                self.N += 1;
                return;
            }

            if (self.vertices.?.contains(n) == false) {
                var node = try self.allocator.create(Node);
                errdefer self.allocator.destroy(node);
                node.* = Node.init(n, d);

                _ = try self.vertices.?.put(node.id, node);
                _ = try self.graph.?.put(node, std.ArrayList(*Edge).init(self.allocator));
                self.N += 1;
            }
        }

        pub fn remove_vertex(self: *Self, n: IdType) ?DType {
            var node_data: ?DType = null;
            if (self.vertices.?.contains(n)) {
                var node: *Node = self.vertices.?.get(n).?;
                node_data = node.data;
                _ = self.vertices.?.remove(n);
                var edges: std.ArrayList(*Edge) = self.graph.?.get(node).?;
                for (edges.items) |edge| {
                    self.allocator.destroy(edge);
                }
                edges.deinit();
                _ = self.graph.?.remove(node);
                self.allocator.destroy(node);
            }
            return node_data;
        }

        pub fn add_edge(self: *Self, n1: IdType, d1: DType, n2: IdType, d2: DType, w: WeightType) !void {
            if (self.N == 0 or self.vertices.?.contains(n1) == false) {
                try self.add_vertex(n1, d1);
            }

            if (self.vertices.?.contains(n2) == false) {
                try self.add_vertex(n2, d2);
            }

            var node1: *Node = self.vertices.?.get(n1).?;
            var node2: *Node = self.vertices.?.get(n2).?;

            var arr: std.ArrayList(*Edge) = self.graph.?.get(node1).?;

            var edge = try self.allocator.create(Edge);
            errdefer self.allocator.destroy(edge);
            edge.* = Edge.init(node2, w);

            try arr.append(edge);

            _ = try self.graph.?.put(node1, arr);
        }

        pub fn print(self: *Self) void {
            std.debug.print("\r\n", .{});
            std.debug.print("Size: {d}\r\n", .{self.N});
            std.debug.print("\r\n", .{});
            std.debug.print("Root: {any}\r\n", .{self.root});
            std.debug.print("\r\n", .{});
            std.debug.print("Vertices:\r\n", .{});
            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                std.debug.print("\r\n{any}\r\n", .{entry.key_ptr.*});
            }
            std.debug.print("\r\n", .{});
            std.debug.print("Graph:\r\n", .{});
            var graph_it = self.graph.?.iterator();
            while (graph_it.next()) |entry| {
                std.debug.print("\r\nConnections: {any}  =>", .{entry.key_ptr.*});
                for (entry.value_ptr.*.items) |v| {
                    std.debug.print("  {}  =>", .{v.*});
                }
                std.debug.print("|| \r\n", .{});
            }
            std.debug.print("\r\n", .{});
        }

        fn topo_driver(self: *Self, node: IdType, comptime T: type, visited: T, stack: *std.ArrayList(*Node)) !bool {
            // In the process of visiting this vertex, we reach the same vertex again.
            // Return to stop the process. (#cond1)
            if (visited.get(node).? == 1) {
                return false;
            }

            // Finished visiting this vertex, it is now marked 2. (#cond2)
            if (visited.get(node).? == 2) {
                return true;
            }

            // Color the node 1, indicating that it is being processed, and initiate a loop
            // to visit all its neighbors. If we reach the same vertex again, return (#cond1)
            _ = try visited.put(node, 1);

            var nodePtr: *Node = self.vertices.?.get(node).?;
            var neighbors: std.ArrayList(*Edge) = self.graph.?.get(nodePtr).?;
            for (neighbors.items) |n| {
                if (visited.get(n.node.id).? == 0) {
                    var check: bool = self.topo_driver(n.node.id, T, visited, stack) catch unreachable;
                    if (check == false) {
                        return false;
                    }
                }
            }

            // Finish processing the current node and mark it 2.
            _ = try visited.put(node, 2);

            // Add node to stack of visited nodes.
            try stack.append(nodePtr);
            return true;
        }

        pub fn topo_sort(self: *Self) !std.ArrayList(*Node) {
            comptime var T: type = if (IdType == []const u8) *std.StringHashMap(i32) else *std.AutoHashMap(IdType, i32);
            var visited = if (IdType == []const u8) std.StringHashMap(i32).init(self.allocator) else std.AutoHashMap(IdType, i32).init(self.allocator);
            defer visited.deinit();

            var stack = std.ArrayList(*Node).init(self.allocator);
            defer stack.deinit();

            var result = std.ArrayList(*Node).init(self.allocator);

            // Initially, color all the nodes 0, to mark them unvisited.
            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                _ = try visited.put(entry.key_ptr.*, 0);
            }
            vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                if (visited.get(entry.key_ptr.*).? == 0) {
                    var check: bool = self.topo_driver(entry.key_ptr.*, T, &visited, &stack) catch unreachable;
                    if (check == false) {
                        for (stack.items) |n| {
                            try result.append(n);
                        }
                        return result;
                    }
                    self.connected += 1;
                }
            }

            self.connected -= 1;

            for (stack.items) |n| {
                try result.append(n);
            }
            return result;
        }

        pub fn dfs(self: *Self) !std.ArrayList(*Node) {
            var visited = if (IdType == []const u8) std.StringHashMap(i32).init(self.allocator) else std.AutoHashMap(IdType, i32).init(self.allocator);
            defer visited.deinit();

            var result = std.ArrayList(*Node).init(self.allocator);

            // Initially, color all the nodes 0, to mark them unvisited.
            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                _ = try visited.put(entry.key_ptr.*, 0);
            }

            var stack = std.ArrayList(*Node).init(self.allocator);
            defer stack.deinit();

            try stack.append(self.root.?);

            while (stack.items.len > 0) {
                var current: *Node = stack.pop();

                var neighbors: std.ArrayList(*Edge) = self.graph.?.get(current).?;
                for (neighbors.items) |n| {
                    if (visited.get(n.node.id).? == 0) {
                        try stack.append(n.node);
                        _ = try visited.put(n.node.id, 1);
                        try result.append(n.node);
                    }
                }
            }

            return result;
        }

        pub fn bfs(self: *Self) !std.ArrayList(*Node) {
            var visited = if (IdType == []const u8) std.StringHashMap(i32).init(self.allocator) else std.AutoHashMap(IdType, i32).init(self.allocator);
            defer visited.deinit();

            var result = std.ArrayList(*Node).init(self.allocator);

            // Initially, color all the nodes 0, to mark them unvisited.
            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                _ = try visited.put(entry.key_ptr.*, 0);
            }

            var qu = std.ArrayList(*Node).init(self.allocator);
            defer qu.deinit();

            try qu.append(self.root.?);

            while (qu.items.len > 0) {
                var current: *Node = qu.orderedRemove(0);

                var neighbors: std.ArrayList(*Edge) = self.graph.?.get(current).?;
                for (neighbors.items) |n| {
                    if (visited.get(n.node.id).? == 0) {
                        try qu.append(n.node);
                        _ = try visited.put(n.node.id, 1);
                        try result.append(n.node);
                    }
                }
            }

            return result;
        }

        pub const Element = struct { id: IdType, distance: WeightType };

        pub fn min_compare(context: void, a: Element, b: Element) std.math.Order {
            _ = context;
            return std.math.order(a.distance, b.distance);
        }

        pub fn dijikstra_shortest_path(self: *Self, src: IdType) !*Shortest {
            if ((self.vertices.?.contains(src) == false)) {
                return error.VertexNotFound;
            }

            var source: *Node = self.vertices.?.get(src).?;

            var pq = std.PriorityQueue(Element, void, min_compare).init(self.allocator, {});
            defer pq.deinit();

            var visited = if (IdType == []const u8) std.StringHashMap(i32).init(self.allocator) else std.AutoHashMap(IdType, i32).init(self.allocator);
            defer visited.deinit();

            var distances = if (IdType == []const u8) std.StringHashMap(WeightType).init(self.allocator) else std.AutoHashMap(IdType, WeightType).init(self.allocator);
            var prev = if (IdType == []const u8) std.StringHashMap(*Node).init(self.allocator) else std.AutoHashMap(IdType, *Node).init(self.allocator);

            // Initially, push all the nodes into the distances hashmap with a distance of infinity.
            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                const equality = if (IdType == []const u8) std.mem.eql(u8, source.id, entry.key_ptr.*) else std.meta.eql(entry.key_ptr.*, src);
                if (!equality) {
                    _ = try distances.put(entry.key_ptr.*, std.math.maxInt(i32));
                    try pq.add(Element{ .id = entry.key_ptr.*, .distance = std.math.maxInt(i32) });
                }
            }

            _ = try distances.put(src, 0);
            try pq.add(Element{ .id = source.id, .distance = 0 });

            while (pq.count() > 0) {
                var current: Element = pq.remove();

                if (!visited.contains(current.id)) {
                    var currentPtr: *Node = self.vertices.?.get(current.id).?;
                    var neighbors: std.ArrayList(*Edge) = self.graph.?.get(currentPtr).?;

                    for (neighbors.items) |n| {
                        // Update the distance values from all neighbors, to the current node
                        // and obtain the shortest distance to the current node from all of its neighbors.
                        var best_dist = distances.get(n.node.id).?;
                        var n_dist = current.distance + n.weight;

                        if (n_dist < best_dist) {
                            // Shortest way to reach current node is through this neighbor.
                            // Update the node's distance from source, and add it to prev.
                            _ = try distances.put(n.node.id, n_dist);

                            _ = try prev.put(n.node.id, currentPtr);

                            // Update the priority queue with the new, shorter distance.
                            var modIndex: usize = 0;
                            for (pq.items, 0..) |item, i| {
                                var equality = if (IdType == []const u8) std.mem.eql(u8, item.id, n.node.id) else std.meta.eql(item.id, n.node.id);
                                if (equality) {
                                    modIndex = i;
                                    break;
                                }
                            }
                            _ = pq.removeIndex(modIndex);
                            try pq.add(Element{ .id = n.node.id, .distance = n_dist });
                        }
                    }
                    _ = try visited.put(current.id, 1);
                }
            }

            return Shortest.init(self.allocator, source, distances, prev, self.vertices);
        }

        fn arrayContains(arr: std.ArrayList(*Node), node: *Node) bool {
            for (arr.items) |item| {
                const equality = if (IdType == []const u8) std.mem.eql(u8, item.id, node.id) else std.meta.eql(item.id, node.id);
                if (equality) {
                    return true;
                }
            }
            return false;
        }

        fn min(a: i32, b: i32) i32 {
            if (a < b) {
                return a;
            }
            return b;
        }

        fn tarjan_driver(self: *Self, current: *Node, globalIndexCounter: *i32, comptime T: type, index: T, low: T, stack: *std.ArrayList(*Node), result: *std.ArrayList(std.ArrayList(*Node))) !void {
            // Set the indices for the current recursion, increment the global index, mark the index
            // for the node, mark low, and append the node to the recursion stack.
            _ = try index.put(current.id, globalIndexCounter.*);
            _ = try low.put(current.id, globalIndexCounter.*);
            try stack.append(current);
            globalIndexCounter.* += 1;

            // Get the neighbors of the current node.
            var neighbors: std.ArrayList(*Edge) = self.graph.?.get(current).?;

            for (neighbors.items) |n| {
                if (index.contains(n.node.id) == false) {
                    self.tarjan_driver(n.node, globalIndexCounter, T, index, low, stack, result) catch unreachable;

                    // Update the low index after the recursion, set low index to the min of
                    // prev and current recursive calls.
                    var currLow: i32 = low.get(current.id).?;
                    var nLow: i32 = low.get(n.node.id).?;

                    _ = try low.put(current.id, min(currLow, nLow));
                } else if (arrayContains(stack.*, current)) {

                    // Update the low index after the recursion, set low index to the min of
                    // prev and current recursive calls.
                    var currLow: i32 = low.get(current.id).?;
                    // IMP: notice that 'index' is being used here, not low.
                    var nIndex: i32 = index.get(n.node.id).?;

                    _ = try low.put(current.id, min(currLow, nIndex));
                }
            }

            var currentLow: i32 = low.get(current.id).?;
            var currentIndex: i32 = index.get(current.id).?;
            if (currentLow == currentIndex) {
                var scc = std.ArrayList(*Node).init(self.allocator);

                while (true) {
                    var successor: *Node = stack.pop();
                    try scc.append(successor);
                    const equality = if (IdType == []const u8) std.mem.eql(u8, successor.id, current.id) else std.meta.eql(successor.id, current.id);
                    if (equality) {
                        try result.append(scc);
                        break;
                    }
                }
            }
        }

        pub fn tarjan(self: *Self) !std.ArrayList(std.ArrayList(*Node)) {
            // Tarjan uses dfs in order to traverse a graph, and return all the strongly connected components in it.
            // The algorithm uses two markers called index and low. Index marks the order in which the node has been visited. The
            // count of nodes from the start vertex. The other marker, low, marks the lowest index value
            // seen by the algorithm so far. Once the recursion unwraps, the key of this algorithm
            // is to compare the current stack 'low' (c1) with the previous stack 'low' (c0)
            // while it collapses the stacks. If c1 < c0, then the low for the previous node is updated
            // to low[prev] = c1, if c1 > c0 then we have found a min-cut edge for the graph. These edges
            // separate the strongly connected components from each other.
            var result = std.ArrayList(std.ArrayList(*Node)).init(self.allocator);

            var globalIndexCounter: i32 = 0;

            var stack = std.ArrayList(*Node).init(self.allocator);
            defer stack.deinit();

            comptime var T: type = if (IdType == []const u8) *std.StringHashMap(i32) else *std.AutoHashMap(IdType, i32);
            var index = if (IdType == []const u8) std.StringHashMap(i32).init(self.allocator) else std.AutoHashMap(IdType, i32).init(self.allocator);
            defer index.deinit();

            var low = if (IdType == []const u8) std.StringHashMap(i32).init(self.allocator) else std.AutoHashMap(IdType, i32).init(self.allocator);
            defer low.deinit();
            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                if (index.contains(entry.value_ptr.*.id) == false) {
                    self.tarjan_driver(entry.value_ptr.*, &globalIndexCounter, T, &index, &low, &stack, &result) catch unreachable;
                }
            }

            return result;
        }

        pub fn dijikstra_all_paths(self: *Self, paths: *AllShortest, adjust_by: *Shortest) !void {
            var pq = std.PriorityQueue(Element, void, min_compare).init(self.allocator, {});
            defer pq.deinit();

            for (paths.nodes, 0..) |u, i| {
                try pq.add(Element{ .id = u.id, .distance = 0 });
                while (pq.count() > 0) {
                    var mid: Element = pq.remove();
                    var k = paths.index_of.get(mid.id).?;
                    if (mid.distance < paths.dist[i][k]) {
                        paths.dist[i][k] = mid.distance;
                    }
                    var mnid = mid.id;
                    var edges_to: std.ArrayList(*Edge) = self.graph.?.get(self.vertices.?.get(mnid).?).?;
                    for (edges_to.items) |e| {
                        var v = e.node;
                        var vid = v.id;
                        var j = paths.index_of.get(vid).?;
                        var w = e.weight + adjust_by.weight_to(mnid) - adjust_by.weight_to(vid);
                        if (w < 0) {
                            std.debug.print("dijkstra: negative edge weight", .{});
                            return error.NegativePathWeight;
                        }
                        var joint = paths.dist[i][k] + w;
                        if (joint < paths.dist[i][j]) {
                            try pq.add(Element{ .id = v.id, .distance = joint });
                            var arr = std.ArrayList(usize).init(self.allocator);
                            try arr.append(k);
                            paths.set(i, j, joint, arr);
                        } else if (joint == paths.dist[i][j]) {
                            var arr = std.ArrayList(usize).init(self.allocator);
                            try arr.append(k);
                            try paths.add(i, j, arr);
                        }
                    }
                }
            }
        }

        pub fn johnson_all_paths(self: *Self) !*AllShortest {
            var paths = try AllShortest.init(self.allocator, self.N, false);

            var prng = std.rand.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                try std.os.getrandom(std.mem.asBytes(&seed));
                break :blk seed;
            });
            const rand = prng.random();

            var vertices_it = self.vertices.?.valueIterator();
            var idx: usize = 0;
            while (vertices_it.next()) |node| : (idx += 1) {
                try paths.index_of.put(node.*.id, idx);
                paths.nodes[idx] = node.*;
            }

            var q: IdType = undefined;
            var sign: IdType = if (IdType == []const u8) "" else -1;
            while (true) {
                // choose random node ID until finding one
                // not already in graph
                if (IdType == []const u8) {
                    var id_buffer: []u8 = try self.allocator.alloc(u8, 16);
                    rand.bytes(id_buffer);
                    q = id_buffer;
                } else {
                    q = sign * rand.int(IdType);
                    sign *= -1;
                }
                if (paths.index_of.get(q) == null) {
                    break;
                }
            }
            var data: DType = undefined;
            for (paths.nodes) |n| {
                try self.add_edge(q, data, n.id, n.data, 0);
            }

            var adjust_by = try self.moore_bellman_ford(q);
            defer adjust_by.deinit();
            _ = self.remove_vertex(q);

            try self.dijikstra_all_paths(paths, adjust_by);

            for (paths.nodes, 0..) |u, i| {
                var hu = adjust_by.weight_to(u.id);
                for (paths.nodes, 0..) |v, j| {
                    if (i == j) {
                        continue;
                    }
                    var hv = adjust_by.weight_to(v.id);
                    paths.dist[i][j] = paths.dist[i][j] - hu + hv;
                }
            }

            return paths;
        }

        pub fn floyd_warshall(self: *Self) !*AllShortest {
            var paths = try AllShortest.init(self.allocator, self.N, true);

            var vertices_it = self.vertices.?.valueIterator();
            var idx: usize = 0;
            while (vertices_it.next()) |node| : (idx += 1) {
                try paths.index_of.put(node.*.id, idx);
                paths.nodes[idx] = node.*;
            }

            for (paths.nodes, 0..) |n, i| {
                paths.dist[i][i] = 0;
                var edges: std.ArrayList(*Edge) = self.graph.?.get(n).?;
                for (edges.items) |e| {
                    var dst_id: IdType = e.node.id;
                    var j: usize = paths.index_of.get(dst_id).?;
                    var p = std.ArrayList(usize).init(self.allocator);
                    try p.append(j);
                    paths.set(i, j, e.weight, p);
                }
            }

            for (paths.nodes, 0..) |_, k| {
                for (paths.nodes, 0..) |_, i| {
                    for (paths.nodes, 0..) |_, j| {
                        var ij = paths.dist[i][j];
                        var joint = paths.dist[i][k] + paths.dist[k][j];
                        if (ij > joint) {
                            paths.set(i, j, joint, try paths.at(i, k));
                        } else if (ij - joint == 0) {
                            try paths.add(i, j, try paths.at(i, k));
                        }
                    }
                }
            }

            var ok = true;
            for (paths.nodes, 0..) |_, i| {
                if (paths.dist[i][i] < 0) {
                    ok = false;
                    break;
                }
            }

            if (!ok) {
                for (paths.nodes, 0..) |_, i| {
                    for (paths.nodes, 0..) |_, j| {
                        for (paths.nodes, 0..) |_, k| {
                            if (std.math.isInf(paths.dist[i][k]) or std.math.isInf(paths.dist[k][j])) {
                                continue;
                            }
                            if (paths.dist[k][k] < 0) {
                                paths.dist[k][k] = -Inf_Val;
                                paths.dist[i][j] = -Inf_Val;
                            }
                        }
                    }
                }
            }

            return paths;
        }

        pub fn moore_bellman_ford(self: *Self, src: IdType) !*Shortest {
            if ((self.vertices.?.contains(src) == false)) {
                return error.VertexNotFound;
            }

            var source: *Node = self.vertices.?.get(src).?;

            var distances = if (IdType == []const u8) std.StringHashMap(WeightType).init(self.allocator) else std.AutoHashMap(IdType, WeightType).init(self.allocator);
            var prev = if (IdType == []const u8) std.StringHashMap(*Node).init(self.allocator) else std.AutoHashMap(IdType, *Node).init(self.allocator);
            var queue = try FifoQueue(IdType, Node).init(self.allocator);
            defer queue.deinit();

            // Initially, push all the nodes into the distances hashmap with a distance of infinity.
            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                const equality = if (IdType == []const u8) std.mem.eql(u8, source.id, entry.key_ptr.*) else std.meta.eql(entry.key_ptr.*, src);
                if (!equality) {
                    _ = try distances.put(entry.key_ptr.*, Inf_Val);
                }
            }
            // initialize src distance to `0`
            _ = try distances.put(src, 0);
            try queue.push(source);

            while (!queue.is_empty()) {
                var src_node: *Node = queue.pop().?;
                var edges: std.ArrayList(*Edge) = self.graph.?.get(src_node).?;
                for (edges.items) |e| {
                    var source_dist: WeightType = distances.get(src_node.id).?;
                    if (source_dist + e.weight < distances.get(e.node.id).?) {
                        try distances.put(e.node.id, source_dist + e.weight);
                        try prev.put(e.node.id, self.vertices.?.get(src_node.*.id).?);
                        try queue.push(e.node);
                    }
                }
            }

            // check for negative cycles
            var edge_it = self.graph.?.iterator();
            while (edge_it.next()) |entry| {
                var source_id: IdType = entry.key_ptr.*.id;
                var edges: std.ArrayList(*Edge) = entry.value_ptr.*;
                for (edges.items) |e| {
                    var source_dist: WeightType = distances.get(source_id).?;
                    var dst_dist: WeightType = distances.get(e.node.id).?;
                    if (source_dist + e.weight < dst_dist) {
                        std.debug.print("Graph contains negative weight cycle", .{});
                        return GraphError.NegativeWeightCycle;
                    }
                }
            }

            return Shortest.init(self.allocator, source, distances, prev, self.vertices);
        }

        // bellman ford algorithm
        pub fn bellman_ford(self: *Self, src: IdType) !*Shortest {
            if ((self.vertices.?.contains(src) == false)) {
                return error.VertexNotFound;
            }

            var source: *Node = self.vertices.?.get(src).?;

            var distances = if (IdType == []const u8) std.StringHashMap(WeightType).init(self.allocator) else std.AutoHashMap(IdType, WeightType).init(self.allocator);
            var prev = if (IdType == []const u8) std.StringHashMap(*Node).init(self.allocator) else std.AutoHashMap(IdType, *Node).init(self.allocator);

            // Initially, push all the nodes into the distances hashmap with a distance of infinity.
            var vertices_it = self.vertices.?.iterator();
            while (vertices_it.next()) |entry| {
                const equality = if (IdType == []const u8) std.mem.eql(u8, source.id, entry.key_ptr.*) else std.meta.eql(entry.key_ptr.*, src);
                if (!equality) {
                    _ = try distances.put(entry.key_ptr.*, Inf_Val);
                }
            }

            // init distance to `src` as `0`
            _ = try distances.put(src, 0);

            // repeat N - 1 times
            var i: usize = 0;
            while (i < self.N - 1) : (i += 1) {
                var edge_it = self.graph.?.iterator();
                while (edge_it.next()) |entry| {
                    var source_id: IdType = entry.key_ptr.*.id;
                    var edges: std.ArrayList(*Edge) = entry.value_ptr.*;
                    for (edges.items) |e| {
                        var source_dist: WeightType = distances.get(source_id).?;
                        if (source_dist + e.weight < distances.get(e.node.id).?) {
                            try distances.put(e.node.id, source_dist + e.weight);
                            try prev.put(e.node.id, self.vertices.?.get(source_id).?);
                        }
                    }
                }
            }

            // check for negative cycles
            var edge_it = self.graph.?.iterator();
            while (edge_it.next()) |entry| {
                var source_id: IdType = entry.key_ptr.*.id;
                var edges: std.ArrayList(*Edge) = entry.value_ptr.*;
                for (edges.items) |e| {
                    var source_dist: WeightType = distances.get(source_id).?;
                    var dst_dist: WeightType = distances.get(e.node.id).?;
                    if (source_dist + e.weight < dst_dist) {
                        std.debug.print("Graph contains negative weight cycle", .{});
                        return GraphError.NegativeWeightCycle;
                    }
                }
            }

            return Shortest.init(self.allocator, source, distances, prev, self.vertices);
        }
    };
}

test "basic graph insertion" {
    const allocator = std.testing.allocator;
    var graph = Graph(i64, i32, f64).init(allocator);
    defer graph.deinit();

    try graph.add_edge(1, 10, 2, 20, 1);
    try graph.add_edge(2, 20, 3, 40, 2);
    try graph.add_edge(3, 110, 1, 10, 3);
    try graph.add_edge(1, 10, 1, 10, 0);
    try graph.add_edge(4, 1, 5, 1, 1);
}

test "basic graph toposort" {
    const allocator = std.testing.allocator;
    var graph = Graph(i64, i32, f64).init(allocator);
    defer graph.deinit();

    try graph.add_edge(1, 10, 2, 20, 1);
    try graph.add_edge(2, 20, 3, 40, 2);
    try graph.add_edge(3, 110, 1, 10, 3);
    try graph.add_edge(1, 10, 1, 10, 0);
    try graph.add_edge(4, 1, 5, 1, 1);

    var res = try graph.topo_sort();
    defer res.deinit();
}

test "basic graph bfs" {
    const allocator = std.testing.allocator;
    var graph = Graph(i64, i32, f64).init(allocator);
    defer graph.deinit();

    try graph.add_edge(1, 10, 2, 20, 1);
    try graph.add_edge(2, 20, 3, 40, 2);
    try graph.add_edge(3, 110, 1, 10, 3);
    try graph.add_edge(1, 10, 1, 10, 0);
    try graph.add_edge(4, 1, 5, 1, 1);

    var res1 = try graph.bfs();
    defer res1.deinit();
}

test "basic graph dfs" {
    const allocator = std.testing.allocator;
    var graph = Graph(i64, i32, f64).init(allocator);
    defer graph.deinit();

    try graph.add_edge(1, 10, 2, 20, 1);
    try graph.add_edge(2, 20, 3, 40, 2);
    try graph.add_edge(3, 110, 1, 10, 3);
    try graph.add_edge(1, 10, 1, 10, 0);
    try graph.add_edge(4, 1, 5, 1, 1);

    var res1 = try graph.dfs();
    defer res1.deinit();
}

test "basic graph dijikstra_shortest_path" {
    // Graph with no self loops for dijiksta.
    const allocator = std.testing.allocator;
    var graph2 = Graph(i64, void, f64).init(allocator);
    defer graph2.deinit();

    try graph2.add_edge(1, {}, 2, {}, 1);
    try graph2.add_edge(1, {}, 3, {}, -10);
    try graph2.add_edge(3, {}, 5, {}, -2);
    try graph2.add_edge(2, {}, 3, {}, 2);
    try graph2.add_edge(3, {}, 4, {}, 5);
    try graph2.add_edge(4, {}, 5, {}, 4);

    var res = try graph2.topo_sort();
    defer res.deinit();

    var res3 = try graph2.dijikstra_shortest_path(1);
    defer res3.deinit();

    var res4 = try res3.path_to(5);
    defer res4.deinit();
}

test "basic graph tarjan" {
    // Graph for tarjan.
    const allocator = std.testing.allocator;
    var graph4 = Graph([]const u8, i32, f64).init(allocator);
    defer graph4.deinit();

    try graph4.add_edge("A", 1, "B", 1, 1);
    try graph4.add_edge("B", 1, "A", 1, 1);
    try graph4.add_edge("B", 1, "C", 1, 2);
    try graph4.add_edge("C", 1, "B", 1, 1);
    try graph4.add_edge("C", 1, "D", 1, 5);
    try graph4.add_edge("D", 1, "E", 1, 4);
    try graph4.add_edge("B", 1, "E", 1, 1);
    try graph4.add_edge("J", 1, "K", 1, 1);
    try graph4.add_edge("M", 1, "N", 1, 1);

    var res = try graph4.topo_sort();
    defer res.deinit();

    var res5 = try graph4.tarjan();
    defer res5.deinit();

    for (res5.items) |n| {
        n.deinit();
    }
}

test "basic bellman ford" {
    const allocator = std.testing.allocator;
    var graph = Graph(i64, void, f64).init(allocator);
    defer graph.deinit();

    try graph.add_edge(1, {}, 2, {}, 6);
    try graph.add_edge(1, {}, 3, {}, 4);
    try graph.add_edge(1, {}, 4, {}, 5);
    try graph.add_edge(2, {}, 5, {}, -1);
    try graph.add_edge(3, {}, 2, {}, -2);
    try graph.add_edge(3, {}, 5, {}, 3);
    try graph.add_edge(4, {}, 3, {}, -2);
    try graph.add_edge(4, {}, 6, {}, -1);
    try graph.add_edge(5, {}, 6, {}, 3);

    var res = try graph.topo_sort();
    defer res.deinit();

    var res5 = try graph.bellman_ford(1);
    defer res5.deinit();

    var path = try res5.path_to(2);
    defer path.deinit();
}

test "basic moore bellman ford" {
    const allocator = std.testing.allocator;
    var graph = Graph(i64, void, f64).init(allocator);
    defer graph.deinit();

    try graph.add_edge(1, {}, 2, {}, 6);
    try graph.add_edge(1, {}, 3, {}, 4);
    try graph.add_edge(1, {}, 4, {}, 5);
    try graph.add_edge(2, {}, 5, {}, -1);
    try graph.add_edge(3, {}, 2, {}, -2);
    try graph.add_edge(3, {}, 5, {}, 3);
    try graph.add_edge(4, {}, 3, {}, -2);
    try graph.add_edge(4, {}, 6, {}, -1);
    try graph.add_edge(5, {}, 6, {}, 3);

    var res = try graph.topo_sort();
    defer res.deinit();

    var res5 = try graph.moore_bellman_ford(1);
    defer res5.deinit();

    var path = try res5.path_to(2);
    defer path.deinit();
}

test "basic floyd warshall" {
    const allocator = std.testing.allocator;
    var graph = Graph(i64, void, f64).init(allocator);
    defer graph.deinit();

    try graph.add_edge(1, {}, 4, {}, 10);
    try graph.add_edge(1, {}, 2, {}, 5);
    try graph.add_edge(2, {}, 3, {}, 3);
    try graph.add_edge(3, {}, 4, {}, 1);

    var res = try graph.topo_sort();
    defer res.deinit();

    var paths = try graph.floyd_warshall();
    defer paths.deinit();

    var path = try paths.between(1, 3);
    defer path.deinit();
}

test "basic johnson" {
    const allocator = std.testing.allocator;
    var graph = Graph(i64, void, f64).init(allocator);
    defer graph.deinit();

    try graph.add_edge(1, {}, 2, {}, 6);
    try graph.add_edge(1, {}, 3, {}, 4);
    try graph.add_edge(1, {}, 4, {}, 5);
    try graph.add_edge(2, {}, 5, {}, -1);
    try graph.add_edge(3, {}, 2, {}, -2);
    try graph.add_edge(3, {}, 5, {}, 3);
    try graph.add_edge(4, {}, 3, {}, -2);
    try graph.add_edge(4, {}, 6, {}, -1);
    try graph.add_edge(5, {}, 6, {}, 3);

    var res = try graph.topo_sort();
    defer res.deinit();

    var paths = try graph.johnson_all_paths();
    defer paths.deinit();

    var path = try paths.between(3, 6);
    defer path.deinit();

    var all_paths = try paths.all_between(1, 6);
    defer all_paths.deinit();
}
