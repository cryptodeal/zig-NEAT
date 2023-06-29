const std = @import("std");
const NNode = @import("../network/nnode.zig").NNode;

pub const MIMOControlGene = struct {
    // current innovation number for gene
    innovation_num: i64,
    // represents impact of mutation on gene
    mutation_num: f64,
    // if true, gene is enabled
    is_enabled: bool,
    // control node with control/activation function
    control_node: *NNode,
    // list of associated IO nodes for fast traversal
    io_nodes: []*NNode,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, control_node: *NNode, innov_num: i64, mut_num: f64, enabled: bool) !*MIMOControlGene {
        var gene: *MIMOControlGene = try allocator.create(MIMOControlGene);
        var io_nodes = std.ArrayList(*NNode).init(allocator);
        for (control_node.incoming.items) |l| {
            try io_nodes.append(l.in_node.?);
        }
        for (control_node.outgoing.items) |l| {
            try io_nodes.append(l.out_node.?);
        }
        gene.* = .{
            .allocator = allocator,
            .control_node = control_node,
            .innovation_num = innov_num,
            .mutation_num = mut_num,
            .is_enabled = enabled,
            .io_nodes = try io_nodes.toOwnedSlice(),
        };
        return gene;
    }

    pub fn init_from_copy(allocator: std.mem.Allocator, g: *MIMOControlGene, control_node: *NNode) !*MIMOControlGene {
        return try MIMOControlGene.init(allocator, control_node, g.innovation_num, g.mutation_num, g.is_enabled);
    }

    pub fn deinit(self: *MIMOControlGene) void {
        for (self.control_node.incoming.items) |l| {
            l.deinit();
        }
        for (self.control_node.outgoing.items) |l| {
            l.deinit();
        }
        self.control_node.deinit();
        self.allocator.free(self.io_nodes);
        self.allocator.destroy(self);
    }

    pub fn is_equal(self: *MIMOControlGene, m: *MIMOControlGene) bool {
        if (self.innovation_num != m.innovation_num or self.mutation_num != m.mutation_num or self.is_enabled != m.is_enabled) {
            return false;
        }
        // TODO: debug this crash/validate node fully
        // if (!self.control_node.is_equal(m.control_node)) {
        if (self.control_node.id != m.control_node.id) {
            return false;
        }
        for (self.io_nodes, m.io_nodes) |a, b| {
            if (!a.is_equal(b)) {
                return false;
            }
        }
        return true;
    }

    pub fn has_intersection(self: *MIMOControlGene, nodes: *std.AutoHashMap(i64, *NNode)) bool {
        for (self.io_nodes) |nd| {
            if (nodes.contains(nd.id)) {
                // found
                return true;
            }
        }
        return false;
    }

    pub fn format(value: MIMOControlGene, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        var enabled_str: []const u8 = "";
        if (!value.is_enabled) {
            enabled_str = " -DISABLED-";
        }
        return writer.print("[MIMO Gene INNOV ({d}, {d:.3}) {s} control node: {any}]", .{ value.innovation_num, value.mutation_num, enabled_str, value.control_node });
    }
};
