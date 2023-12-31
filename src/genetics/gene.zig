const std = @import("std");
const Link = @import("../network/link.zig").Link;
const NNode = @import("../network/nnode.zig").NNode;
const Trait = @import("../trait.zig").Trait;
const neat_math = @import("../math/activations.zig");
const NodeNeuronType = @import("../network/common.zig").NodeNeuronType;
const traitWithId = @import("common.zig").traitWithId;

/// Encoded Gene for (de)serialization to/from JSON.
pub const GeneJSON = struct {
    src_id: i64,
    tgt_id: i64,
    weight: f64,
    trait_id: ?i64,
    innov_num: i64,
    mut_num: f64,
    recurrent: bool,
    enabled: bool,
};

/// The Gene type in this system specifies a "Connection Gene."
/// Nodes are represented using the NNode struct, which serves as both a genotypic and phenotypic representation of nodes.
/// Genetic Representation of connections uses this special class because it calls for special operations better served
/// by a specific genetic representation.
/// A Gene object in this system specifies a link between two nodes along with an `innovation_num` which tells when
/// in the history of a population the gene first arose. This allows the system to track innovations and use those to
/// determine which organisms are compatible (i.e. in the same species).
/// A `mutation_num` gives a rough sense of how much mutation the gene has experienced since it originally appeared
/// (Since it was first innovated). In the current implementation the mutation number is the same as the weight.
pub const Gene = struct {
    /// The link between nodes.
    link: *Link,
    /// The current innovation number for this gene.
    innovation_num: i64,
    /// Represents the extent to which mutation has changed the link.
    mutation_num: f64,
    /// If true, the gene is enabled.
    is_enabled: bool,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// For internal use only; parses Gene into format that can be serialized to JSON.
    pub fn jsonify(self: *Gene) GeneJSON {
        return .{
            .src_id = self.link.in_node.?.id,
            .tgt_id = self.link.out_node.?.id,
            .weight = self.link.cxn_weight,
            .trait_id = if (self.link.trait != null) self.link.trait.?.id.? else null,
            .innov_num = self.innovation_num,
            .mut_num = self.mutation_num,
            .recurrent = self.link.is_recurrent,
            .enabled = self.is_enabled,
        };
    }

    /// Initializes a new Gene from JSON (used when reading Genome from file).
    pub fn initFromJSON(allocator: std.mem.Allocator, value: GeneJSON, traits: []*Trait, nodes: []*NNode) !*Gene {
        var trait = traitWithId(if (value.trait_id == null) 0 else value.trait_id.?, traits);
        var innov_num = value.innov_num;
        var weight = value.weight;
        var mut_num = value.mut_num;
        var enabled = value.enabled;
        var recurrent = value.recurrent;
        var in_node: *NNode = undefined;
        var out_node: *NNode = undefined;
        for (nodes) |node| {
            if (node.id == value.src_id) in_node = node;
            if (node.id == value.tgt_id) out_node = node;
        }
        if (trait != null) {
            return Gene.initConnectionGene(allocator, try Link.initWithTrait(allocator, trait, weight, in_node, out_node, recurrent), innov_num, mut_num, enabled);
        } else {
            return Gene.initConnectionGene(allocator, try Link.init(allocator, weight, in_node, out_node, recurrent), innov_num, mut_num, enabled);
        }
    }

    /// Initializes a new Gene.
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

    /// Initializes a new Gene with a Trait.
    pub fn initWithTrait(allocator: std.mem.Allocator, trait: ?*Trait, weight: f64, in_node: ?*NNode, out_node: ?*NNode, recurrent: bool, innovation_num: i64, mutation_num: f64) !*Gene {
        var gene: *Gene = try allocator.create(Gene);
        gene.* = .{
            .allocator = allocator,
            .link = try Link.initWithTrait(allocator, trait, weight, in_node, out_node, recurrent),
            .innovation_num = innovation_num,
            .mutation_num = mutation_num,
            .is_enabled = true,
        };
        return gene;
    }

    /// Initializes a new Gene by copying an existing Gene.
    pub fn initCopy(allocator: std.mem.Allocator, gene: *Gene, trait: ?*Trait, in_node: ?*NNode, out_node: ?*NNode) !*Gene {
        var new_gene: *Gene = try allocator.create(Gene);
        new_gene.* = .{
            .allocator = allocator,
            .link = try Link.initWithTrait(allocator, trait, gene.link.cxn_weight, in_node, out_node, gene.link.is_recurrent),
            .innovation_num = gene.innovation_num,
            .mutation_num = gene.mutation_num,
            .is_enabled = true,
        };
        return new_gene;
    }

    /// Initializes a new Gene with the provided Link.
    pub fn initConnectionGene(allocator: std.mem.Allocator, link: *Link, innovation_num: i64, mutation_num: f64, enabled: bool) !*Gene {
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

    /// Initializes a new Gene from file (used when reading Genome from plain text file).
    pub fn readFromFile(allocator: std.mem.Allocator, data: []const u8, traits: []*Trait, nodes: []*NNode) !*Gene {
        var trait_id: i64 = undefined;
        var in_node_id: i64 = undefined;
        var out_node_id: i64 = undefined;
        var innovation_num: i64 = undefined;
        var weight: f64 = undefined;
        var mut_num: f64 = undefined;
        var recurrent: bool = false;
        var enabled: bool = false;

        var split = std.mem.split(u8, data, " ");

        var count: usize = 0;
        while (split.next()) |d| : (count += 1) {
            switch (count) {
                0 => trait_id = try std.fmt.parseInt(i64, d, 10),
                1 => in_node_id = try std.fmt.parseInt(i64, d, 10),
                2 => out_node_id = try std.fmt.parseInt(i64, d, 10),
                3 => weight = try std.fmt.parseFloat(f64, d),
                4 => {
                    if (std.mem.eql(u8, d, "true")) recurrent = true;
                },
                5 => innovation_num = try std.fmt.parseInt(i64, d, 10),
                6 => mut_num = try std.fmt.parseFloat(f64, d),
                7 => {
                    if (std.mem.eql(u8, d, "true")) enabled = true;
                },
                else => break,
            }
        }
        if (count < 7) return error.MalformedGeneInGenomeFile;
        var in_node: ?*NNode = null;
        var out_node: ?*NNode = null;
        for (nodes) |nd| {
            if (nd.id == in_node_id) in_node = nd;
            if (nd.id == out_node_id) out_node = nd;
        }
        var trait = traitWithId(trait_id, traits);
        if (trait != null) {
            var link = try Link.initWithTrait(allocator, trait, weight, in_node, out_node, recurrent);
            return Gene.initConnectionGene(allocator, link, innovation_num, mut_num, enabled);
        } else {
            var link = try Link.init(allocator, weight, in_node, out_node, recurrent);
            return Gene.initConnectionGene(allocator, link, innovation_num, mut_num, enabled);
        }
    }

    /// Frees all associated memory.
    pub fn deinit(self: *Gene) void {
        self.link.deinit();
        self.allocator.destroy(self);
    }

    /// Tests equality of Gene against another Gene.
    pub fn isEql(self: *Gene, g: *Gene) bool {
        if (self.innovation_num != g.innovation_num or self.mutation_num != g.mutation_num or self.is_enabled != g.is_enabled) {
            return false;
        }
        return self.link.isEql(g.link);
    }

    /// Formats Gene for printing to writer.
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
    var node1 = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer node1.deinit();
    node1.activation_type = neat_math.NodeActivationType.NullActivation;
    var node2 = try NNode.init(allocator, 2, NodeNeuronType.OutputNeuron);
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
    var g1 = try Gene.initWithTrait(allocator, trait, 3.2, node1, node2, true, 42, 5.2);
    defer g1.deinit();

    var g_copy = try Gene.initCopy(allocator, g1, trait, node1, node2);
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
