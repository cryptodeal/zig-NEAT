const std = @import("std");
const Link = @import("../network/link.zig").Link;
const NNode = @import("../network/nnode.zig").NNode;
const Trait = @import("../trait.zig").Trait;
const neat_math = @import("../math/activations.zig");
const NodeNeuronType = @import("../network/common.zig").NodeNeuronType;

pub const Gene = struct {
    // link between nodes
    link: *Link,
    // current innovation number for gene
    innovation_num: i64,
    // represents impact of mutation on gene
    mutation_num: f64,
    // if true, gene is enabled
    is_enabled: bool,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, weight: f64, in_node: ?*NNode, out_node: ?*NNode, recurrent: bool, innovation_num: i64, mutation_num: f64) !*Gene {
        var gene: *Gene = try allocator.create(Gene);
        gene.* = .{
            .allocator = allocator,
            .link = try Link.init(allocator, weight, in_node, out_node, recurrent),
            .innovation_num = innovation_num,
            .mutation_num = mutation_num,
            .is_enabled = true,
        };
        return gene;
    }

    pub fn init_with_trait(allocator: std.mem.Allocator, trait: ?*Trait, weight: f64, in_node: ?*NNode, out_node: ?*NNode, recurrent: bool, innovation_num: i64, mutation_num: f64) !*Gene {
        var gene: *Gene = try allocator.create(Gene);
        gene.* = .{
            .allocator = allocator,
            .link = try Link.init_with_trait(allocator, trait, weight, in_node, out_node, recurrent),
            .innovation_num = innovation_num,
            .mutation_num = mutation_num,
            .is_enabled = true,
        };
        return gene;
    }

    pub fn init_copy(allocator: std.mem.Allocator, gene: *Gene, trait: ?*Trait, in_node: ?*NNode, out_node: ?*NNode) !*Gene {
        var new_gene: *Gene = try allocator.create(Gene);
        new_gene.* = .{
            .allocator = allocator,
            .link = try Link.init_with_trait(allocator, trait, gene.link.cxn_weight, in_node, out_node, gene.link.is_recurrent),
            .innovation_num = gene.innovation_num,
            .mutation_num = gene.mutation_num,
            .is_enabled = true,
        };
        return new_gene;
    }

    pub fn init_cxn_gene(allocator: std.mem.Allocator, link: *Link, innovation_num: i64, mutation_num: f64, enabled: bool) !*Gene {
        var self = try allocator.create(Gene);
        self.* = .{
            .allocator = allocator,
            .link = link,
            .innovation_num = innovation_num,
            .mutation_num = mutation_num,
            .is_enabled = enabled,
        };
        return self;
    }

    pub fn deinit(self: *Gene) void {
        self.link.deinit();
        self.allocator.destroy(self);
    }

    pub fn is_equal(self: *Gene, g: *Gene) bool {
        if (self.innovation_num != g.innovation_num or self.mutation_num != g.mutation_num or self.is_enabled != g.is_enabled) {
            return false;
        }
        return self.link.is_equal(g.link);
    }
    pub fn format(value: Gene, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        var enabled_str: []const u8 = "";
        if (!value.is_enabled) {
            enabled_str = " -DISABLED-";
        }
        var recurrent_str: []const u8 = "";
        if (value.link.is_recurrent) {
            recurrent_str = " -RECUR-";
        }

        try writer.print("[Link ({d} ->{d}) INNOV ({d}, {d:.3}) Weight: {d:.3} ", .{ value.link.in_node.?.id, value.link.out_node.?.id, value.innovation_num, value.mutation_num, value.link.cxn_weight });
        if (value.link.trait != null) {
            try writer.print(" Link's trait_id: {any}", .{value.link.trait.?.id});
        }
        return writer.print("{s}{s} : {any}->{any}]", .{ enabled_str, recurrent_str, value.link.in_node, value.link.out_node });
    }
};

test "new Gene copy" {
    var allocator = std.testing.allocator;
    // init nodes
    var node1 = try NNode.new_NNode(allocator, 1, NodeNeuronType.InputNeuron);
    defer node1.deinit();
    node1.activation_type = neat_math.NodeActivationType.NullActivation;
    var node2 = try NNode.new_NNode(allocator, 2, NodeNeuronType.OutputNeuron);
    defer node2.deinit();
    node2.activation_type = neat_math.NodeActivationType.SigmoidSteepenedActivation;

    // init trait
    var trait = try Trait.init(allocator, 8);
    defer trait.deinit();
    trait.id = 1;
    for (trait.params, 0..) |*p, i| {
        p.* = if (i == 0) 0.1 else 0;
    }

    // init gene1
    var g1 = try Gene.init_with_trait(allocator, trait, 3.2, node1, node2, true, 42, 5.2);
    defer g1.deinit();

    var g_copy = try Gene.init_copy(allocator, g1, trait, node1, node2);
    defer g_copy.deinit();

    // check for equality
    try std.testing.expect(node1.id == g_copy.link.in_node.?.id);
    try std.testing.expect(node2.id == g_copy.link.out_node.?.id);
    try std.testing.expect(trait.id == g_copy.link.trait.?.id);
    try std.testing.expectEqual(trait.params, g_copy.link.trait.?.params);
    try std.testing.expect(g1.innovation_num == g_copy.innovation_num);
    try std.testing.expect(g1.link.cxn_weight == g_copy.link.cxn_weight);
    try std.testing.expect(g1.mutation_num == g_copy.mutation_num);
    try std.testing.expect(g1.is_enabled == g_copy.is_enabled);
}
