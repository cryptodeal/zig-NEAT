const std = @import("std");
const Trait = @import("../trait.zig").Trait;
const NNode = @import("../network/nnode.zig").NNode;

pub const InnovationType = enum(u8) {
    NewNodeInnType,
    NewLinkInnType,
};

pub const MutatorType = enum(u8) {
    GaussianMutator,
    GoldGaussianMutator,
};

pub const GenomeEncoding = enum(u8) {
    PlainGenomeEncoding,
    YAMLGenomeEncoding,
};

pub fn trait_with_id(trait_id: i64, traits: ?[]*Trait) ?*Trait {
    if (trait_id != 0 and traits != null) {
        for (traits.?) |tr| {
            if (tr.id.? == trait_id) {
                return tr;
            }
        }
    }
    return null;
}

pub fn node_with_id(node_id: i64, nodes: ?[]*NNode) ?*NNode {
    if (node_id != 0 and nodes != null) {
        for (nodes.?) |n| {
            if (n.id == node_id) {
                return n;
            }
        }
    }
    return null;
}

pub fn genome_encoding_from_file_name(file_name: []const u8) GenomeEncoding {
    if (std.mem.endsWith(u8, file_name, "yml") or std.mem.endsWith(u8, file_name, "yaml")) {
        return GenomeEncoding.YAMLGenomeEncoding;
    } else {
        return GenomeEncoding.PlainGenomeEncoding;
    }
}
