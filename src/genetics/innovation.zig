const std = @import("std");
const InnovationType = @import("common.zig").InnovationType;

pub const Innovation = struct {
    // specify where the innovation occurred
    in_node_id: i64,
    out_node_id: i64,
    // number assigned to the innovation
    innovation_num: i64,
    // if new node innovation, there are 2 innovations (links) added for the new node
    innnovation_num2: i64 = undefined,

    // if link is added, this is its weight
    new_weight: f64 = undefined,
    // if link is added, this is its connected trait index
    new_trait_num: usize = undefined,
    // if new node created, this is its ID
    new_node_id: i64 = undefined,

    // if new node created, this is the innovation number of the gene's link it is being inserted into
    old_innov_num: i64 = undefined,
    // flag indicating whether innovation is for recurrent link
    is_recurrent: bool = false,
    // either new node or new link
    innovation_type: InnovationType,

    allocator: std.mem.Allocator,

    pub fn init_for_node(allocator: std.mem.Allocator, in_node_id: i64, out_node_id: i64, innovation_num_1: i64, innovation_num_2: i64, new_node_id: i64, old_innovation_num: i64) !*Innovation {
        var self = try allocator.create(Innovation);
        self.* = .{
            .allocator = allocator,
            .innovation_type = InnovationType.NewNodeInnType,
            .in_node_id = in_node_id,
            .out_node_id = out_node_id,
            .innovation_num = innovation_num_1,
            .innnovation_num2 = innovation_num_2,
            .new_node_id = new_node_id,
            .old_innov_num = old_innovation_num,
        };
        return self;
    }

    pub fn init_for_link(allocator: std.mem.Allocator, in_node_id: i64, out_node_id: i64, innovation_num: i64, weight: f64, trait_id: usize) !*Innovation {
        var self = try allocator.create(Innovation);
        self.* = .{
            .allocator = allocator,
            .innovation_type = InnovationType.NewLinkInnType,
            .in_node_id = in_node_id,
            .out_node_id = out_node_id,
            .innovation_num = innovation_num,
            .new_weight = weight,
            .new_trait_num = trait_id,
        };
        return self;
    }

    pub fn init_for_recurrent_link(allocator: std.mem.Allocator, in_node_id: i64, out_node_id: i64, innovation_num: i64, weight: f64, trait_id: usize, recur: bool) !*Innovation {
        var self = try allocator.create(Innovation);
        self.* = .{
            .allocator = allocator,
            .innovation_type = InnovationType.NewLinkInnType,
            .in_node_id = in_node_id,
            .out_node_id = out_node_id,
            .innovation_num = innovation_num,
            .new_weight = weight,
            .new_trait_num = trait_id,
            .is_recurrent = recur,
        };
        return self;
    }

    pub fn deinit(self: *Innovation) void {
        self.allocator.destroy(self);
    }
};
