const std = @import("std");
const net_node = @import("nnode.zig");
const net_common = @import("common.zig");
const neat_trait = @import("../trait.zig");

const NodeNeuronType = net_common.NodeNeuronType;
const Trait = neat_trait.Trait;
const NNode = net_node.NNode;

pub const Link = struct {
    // weight of cxn
    cxn_weight: f64,
    // NNode input for link
    in_node: ?*NNode,
    // NNode output for link
    out_node: ?*NNode,
    // if true, link is recurrent
    is_recurrent: bool = false,
    // if true, link is time delayed
    is_time_delayed: bool = false,
    // points to trait of params for genetic creation
    trait: ?*Trait,

    params: []f64,
    has_params: bool = false,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, weight: f64, input_node: ?*NNode, output_node: ?*NNode, recurrent: bool) !*Link {
        var link: *Link = try allocator.create(Link);
        link.* = .{
            .allocator = allocator,
            .cxn_weight = weight,
            .in_node = input_node,
            .out_node = output_node,
            .is_recurrent = recurrent,
            .params = undefined,
            .trait = null,
        };
        return link;
    }

    pub fn init_with_trait(allocator: std.mem.Allocator, trait: ?*Trait, weight: f64, input_node: ?*NNode, output_node: ?*NNode, recurrent: bool) !*Link {
        var link: *Link = try Link.init(allocator, weight, input_node, output_node, recurrent);
        link.trait = trait;
        try link.derive_trait(trait);
        return link;
    }

    pub fn init_copy(allocator: std.mem.Allocator, l: *Link, in_node: *NNode, out_node: *NNode) !*Link {
        var link: *Link = try Link.init(allocator, l.cxn_weight, in_node, out_node, l.is_recurrent);
        link.trait = l.trait;
        try link.derive_trait(l.trait);
        return link;
    }

    pub fn deinit(self: *Link) void {
        if (self.has_params) {
            self.allocator.free(self.params);
        }
        self.allocator.destroy(self);
    }

    pub fn is_equal(self: *Link, l: *Link) bool {
        // check equality of fields w primitive types
        if (self.cxn_weight != l.cxn_weight or self.is_recurrent != l.is_recurrent or self.is_time_delayed != l.is_time_delayed or self.has_params != l.has_params) {
            return false;
        }
        // check trait equality
        if ((self.trait == null and l.trait != null) or (self.trait != null and l.trait == null)) {
            return false;
        } else if (self.trait != null and l.trait != null and !self.trait.?.is_equal(l.trait.?)) {
            return false;
        }
        // check param equality
        if (self.has_params and l.has_params and !std.mem.eql(f64, self.params, l.params)) {
            return false;
        }
        // check in_node equality
        if ((self.in_node == null and l.in_node != null) or (self.in_node != null and l.in_node == null)) {
            return false;
        } else if (self.in_node != null and l.in_node != null and !self.in_node.?.is_equal(l.in_node.?)) {
            return false;
        }
        // check out node equality
        if ((self.out_node == null and l.out_node != null) or (self.out_node != null and l.out_node == null)) {
            return false;
        } else if (self.out_node != null and l.out_node != null and !self.out_node.?.is_equal(l.out_node.?)) {
            return false;
        }
        return true;
    }

    pub fn derive_trait(self: *Link, t: ?*Trait) !void {
        if (t != null) {
            self.params = try self.allocator.alloc(f64, t.?.params.len);
            self.has_params = true;
            for (t.?.params, 0..) |p, i| {
                self.params[i] = p;
            }
        }
    }

    pub fn is_genetically_eql(self: *Link, other: *Link) bool {
        const same_in_node: bool = self.in_node.?.id == other.in_node.?.id;
        // std.debug.print("\nsame_in_node: {any}; self.in_node.id: {d} - other.in_node.id: {d}\n", .{ same_in_node, self.in_node.?.id, other.in_node.?.id });
        const same_out_node: bool = self.out_node.?.id == other.out_node.?.id;
        // std.debug.print("same_out_node: {any}; self.out_node.id: {d} - other.out_node.id: {d}\n", .{ same_out_node, self.out_node.?.id, other.out_node.?.id });
        const same_recurrent: bool = self.is_recurrent == other.is_recurrent;
        // std.debug.print("same_recurrent: {any}; self.is_recurrent: {any} - other.is_recurrent: {any}\n", .{ same_recurrent, self.is_recurrent, other.is_recurrent });
        return same_in_node and same_out_node and same_recurrent;
    }

    pub fn string(self: *Link, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().print("[Link: ({any} <-> {any}), weight: {d:.3}, recurrent: {any}, time delayed: {any}]", .{ self.in_node, self.out_node, self.cxn_weight, self.is_recurrent, self.is_time_delayed });
        return buffer.toOwnedSlice();
    }

    pub fn id_string(self: *Link, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        try buffer.writer().print("{d}-{d}", .{ self.in_node.?.id, self.out_node.?.id });
        return buffer.toOwnedSlice();
    }
};

test "is genetically equal" {
    const allocator = std.testing.allocator;
    var in = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer in.deinit();
    var out = try NNode.init(allocator, 2, NodeNeuronType.OutputNeuron);
    defer out.deinit();

    var link1 = try Link.init(allocator, 1.0, in, out, false);
    defer link1.deinit();
    var link2 = try Link.init(allocator, 2.0, in, out, false);
    defer link2.deinit();

    var equals = link1.is_genetically_eql(link2);
    try std.testing.expect(equals);

    var link3 = try Link.init(allocator, 2.0, in, out, true);
    defer link3.deinit();
    equals = link1.is_genetically_eql(link3);
    try std.testing.expect(!equals);

    var hidden = try NNode.init(allocator, 3, NodeNeuronType.HiddenNeuron);
    defer hidden.deinit();
    var link4 = try Link.init(allocator, 2.0, in, hidden, false);
    defer link4.deinit();
    equals = link1.is_genetically_eql(link4);
    try std.testing.expect(!equals);

    var in2 = try NNode.init(allocator, 3, NodeNeuronType.InputNeuron);
    defer in2.deinit();
    var link5 = try Link.init(allocator, 2.0, in2, out, false);
    defer link5.deinit();
    equals = link1.is_genetically_eql(link5);
    try std.testing.expect(!equals);
}

test "new link copy" {
    const allocator = std.testing.allocator;
    var in = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer in.deinit();
    var out = try NNode.init(allocator, 2, NodeNeuronType.OutputNeuron);
    defer out.deinit();

    var trait: *Trait = try Trait.init(allocator, 6);
    defer trait.deinit();
    // manually set trait weights for test
    trait.params[0] = 1.1;
    trait.params[1] = 2.3;
    trait.params[2] = 3.4;
    trait.params[3] = 4.2;
    trait.params[4] = 5.5;
    trait.params[5] = 6.7;

    var link = try Link.init_with_trait(allocator, trait, 1.0, in, out, false);
    defer link.deinit();

    var in_copy = try NNode.init(allocator, 3, NodeNeuronType.InputNeuron);
    defer in_copy.deinit();
    var out_copy = try NNode.init(allocator, 4, NodeNeuronType.HiddenNeuron);
    defer out_copy.deinit();
    var link_copy = try Link.init_copy(allocator, link, in_copy, out_copy);
    defer link_copy.deinit();

    try std.testing.expect(link.cxn_weight == link_copy.cxn_weight);
    try std.testing.expectEqualSlices(f64, link.params, link_copy.params);
    try std.testing.expect(link.is_recurrent == link_copy.is_recurrent);
    try std.testing.expectEqual(in_copy, link_copy.in_node.?);
    try std.testing.expectEqual(out_copy, link_copy.out_node.?);
}

test "new link w trait" {
    const allocator = std.testing.allocator;
    var in = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer in.deinit();
    var out = try NNode.init(allocator, 2, NodeNeuronType.OutputNeuron);
    defer out.deinit();

    const w: f64 = 10.9;

    var trait: *Trait = try Trait.init(allocator, 6);
    defer trait.deinit();
    // manually set trait weights for test
    trait.params[0] = 1.1;
    trait.params[1] = 2.3;
    trait.params[2] = 3.4;
    trait.params[3] = 4.2;
    trait.params[4] = 5.5;
    trait.params[5] = 6.7;

    var link = try Link.init_with_trait(allocator, trait, w, in, out, false);
    defer link.deinit();

    try std.testing.expectEqual(in, link.in_node.?);
    try std.testing.expectEqual(out, link.out_node.?);
    try std.testing.expectEqualSlices(f64, trait.params, link.params);
    try std.testing.expect(link.cxn_weight == w);
    try std.testing.expect(!link.is_recurrent);
}

test "new link" {
    const allocator = std.testing.allocator;
    var in = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer in.deinit();
    var out = try NNode.init(allocator, 2, NodeNeuronType.OutputNeuron);
    defer out.deinit();

    const w: f64 = 10.9;

    var link = try Link.init(allocator, w, in, out, true);
    defer link.deinit();

    try std.testing.expectEqual(in, link.in_node.?);
    try std.testing.expectEqual(out, link.out_node.?);
    try std.testing.expect(link.cxn_weight == w);
    try std.testing.expect(link.is_recurrent);
}

test "stringify link" {
    const allocator = std.testing.allocator;
    var in = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer in.deinit();
    var out = try NNode.init(allocator, 2, NodeNeuronType.OutputNeuron);
    defer out.deinit();

    const w: f64 = 10.9;

    var link = try Link.init(allocator, w, in, out, true);
    defer link.deinit();

    // TODO: fix segmentation fault caused when trying to free allocated memory for string
    const str = try link.string(allocator);
    defer allocator.free(str);
    try std.testing.expect(str.len > 0);
}

test "link id string" {
    const allocator = std.testing.allocator;
    var in = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer in.deinit();
    var out = try NNode.init(allocator, 2, NodeNeuronType.OutputNeuron);
    defer out.deinit();

    var link = try Link.init(allocator, 0, in, out, true);
    defer link.deinit();

    const id_str = try link.id_string(allocator);
    defer allocator.free(id_str);
    try std.testing.expect(id_str.len > 0);
    try std.testing.expectEqualStrings("1-2", id_str);
}
