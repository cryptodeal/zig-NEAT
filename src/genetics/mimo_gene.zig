const std = @import("std");
const neat_math = @import("../math/activations.zig");
const genetics_common = @import("common.zig");

const NNode = @import("../network/nnode.zig").NNode;
const Link = @import("../network/link.zig").Link;
const Trait = @import("../trait.zig").Trait;
const traitWithId = genetics_common.traitWithId;
const nodeWithId = genetics_common.nodeWithId;

/// Encoded MIMOControlGeneLink for (de)serialization to/from JSON.
pub const MIMOControlGeneLinkJSON = struct {
    id: i64,
};

/// Encoded MIMOControlGene for (de)serialization to/from JSON.
pub const MIMOControlGeneJSON = struct {
    id: i64,
    trait_id: ?i64,
    activation: neat_math.NodeActivationType,
    innov_num: i64,
    mut_num: f64,
    enabled: bool,
    inputs: []MIMOControlGeneLinkJSON,
    outputs: []MIMOControlGeneLinkJSON,

    pub fn deinit(self: *MIMOControlGeneJSON, allocator: std.mem.Allocator) void {
        allocator.free(self.inputs);
        allocator.free(self.outputs);
    }
};

/// MIMOControlGene or the Multiple-Input Multiple-Output (MIMO) control Gene, which allows creating modular Genomes,
/// in which several groups of Genes connected through single MIMO Gene and corresponding control function is
/// applied to all inputs in order to produce outputs. This allows building modular hierarchical Genomes which
/// can be considered as sum of constituent components and evolved as a whole and as a concrete parts simultaneously.
pub const MIMOControlGene = struct {
    /// The current innovation number for the Gene.
    innovation_num: i64,
    /// Represents the impact of mutation on the Gene.
    mutation_num: f64,
    /// If true, the Gene is enabled.
    is_enabled: bool,
    /// The control node with control/activation function.
    control_node: *NNode,
    /// The list of associated IO nodes for fast traversal.
    io_nodes: []*NNode,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// For internal use only; parses MIMOControlGene into format that can be serialized to JSON.
    pub fn jsonify(self: *MIMOControlGene, allocator: std.mem.Allocator) !MIMOControlGeneJSON {
        var inputs = try allocator.alloc(MIMOControlGeneLinkJSON, self.control_node.incoming.items.len);
        for (self.control_node.incoming.items, 0..) |l, i| {
            inputs[i] = .{ .id = l.in_node.?.id };
        }
        var outputs = try allocator.alloc(MIMOControlGeneLinkJSON, self.control_node.outgoing.items.len);
        for (self.control_node.outgoing.items, 0..) |l, i| {
            outputs[i] = .{ .id = l.out_node.?.id };
        }
        return .{
            .id = self.control_node.id,
            .trait_id = if (self.control_node.trait != null) self.control_node.trait.?.id.? else null,
            .activation = self.control_node.activation_type,
            .innov_num = self.innovation_num,
            .mut_num = self.mutation_num,
            .enabled = self.is_enabled,
            .inputs = inputs,
            .outputs = outputs,
        };
    }

    /// Initializes a new MIMOControlGene.
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

    /// Initializes a new MIMOControlGene from an existing MIMOControlGene.
    pub fn initCopy(allocator: std.mem.Allocator, g: *MIMOControlGene, control_node: *NNode) !*MIMOControlGene {
        return MIMOControlGene.init(allocator, control_node, g.innovation_num, g.mutation_num, g.is_enabled);
    }

    /// Initializes a new MIMOControlGene from JSON (used when reading Genome from file).
    pub fn initFromJSON(allocator: std.mem.Allocator, value: MIMOControlGeneJSON, traits: []*Trait, nodes: []*NNode) !*MIMOControlGene {
        var control_node = try NNode.init(allocator, value.id, .HiddenNeuron);
        // set activation function
        control_node.activation_type = value.activation;
        var trait = traitWithId(if (value.trait_id == null) 0 else value.trait_id.?, traits);
        if (trait != null) control_node.trait = trait.?;

        // read MIMO gene parameters
        var innov_num = value.innov_num;
        var mut_num = value.mut_num;
        var enabled = value.enabled;

        // read input links
        for (value.inputs) |input| {
            var node = nodeWithId(input.id, nodes);
            if (node != null) {
                try control_node.incoming.append(try Link.init(allocator, 1, node.?, control_node, false));
            } else {
                std.debug.print("no MIMO input node with id: {d} can be found for module: {d}\n", .{ input.id, control_node.id });
                return error.MissingMIMOInputNode;
            }
        }

        // read output links
        for (value.outputs) |output| {
            var node = nodeWithId(output.id, nodes);
            if (node != null) {
                try control_node.outgoing.append(try Link.init(allocator, 1, control_node, node.?, false));
            } else {
                std.debug.print("no MIMO output node with id: {d} can be found for module: {d}\n", .{ output.id, control_node.id });
                return error.MissingMIMOOutputNode;
            }
        }

        return MIMOControlGene.init(allocator, control_node, innov_num, mut_num, enabled);
    }

    /// Frees all associated memory.
    pub fn deinit(self: *MIMOControlGene) void {
        for (self.control_node.incoming.items) |l| l.deinit();
        for (self.control_node.outgoing.items) |l| l.deinit();
        self.control_node.deinit();
        self.allocator.free(self.io_nodes);
        self.allocator.destroy(self);
    }

    /// Tests equality of MIMOControlGene against another MIMOControlGene.
    pub fn isEql(self: *MIMOControlGene, m: *MIMOControlGene) bool {
        if (self.innovation_num != m.innovation_num or self.mutation_num != m.mutation_num or self.is_enabled != m.is_enabled) {
            return false;
        }
        // TODO: debug this crash/validate node fully
        // if (!self.control_node.isEql(m.control_node)) {
        if (self.control_node.id != m.control_node.id) {
            return false;
        }
        for (self.io_nodes, m.io_nodes) |a, b| {
            if (!a.isEql(b)) {
                return false;
            }
        }
        return true;
    }

    /// Tests whether this MIMOControlGene has intersection with provided map of nodes, i.e. any of it's IO nodes included into list.
    pub fn hasIntersection(self: *MIMOControlGene, nodes: *std.AutoHashMap(i64, *NNode)) bool {
        for (self.io_nodes) |nd| {
            if (nodes.contains(nd.id)) {
                // found
                return true;
            }
        }
        return false;
    }

    /// Formats MIMOControlGene for printing to writer.
    pub fn format(value: MIMOControlGene, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        var enabled_str: []const u8 = "";
        if (!value.is_enabled) {
            enabled_str = " -DISABLED-";
        }
        return writer.print("[MIMO Gene INNOV ({d}, {d:.3}) {s} control node: {any}]", .{ value.innovation_num, value.mutation_num, enabled_str, value.control_node });
    }
};
