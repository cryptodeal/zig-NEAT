const std = @import("std");
const InnovationType = @import("common.zig").InnovationType;

/// Innovation serves as a way to record innovations specifically, so that an innovation in one Genome can be
/// compared with other innovations in the same epoch, and if they are the same innovation, they can both be assigned the
/// same innovation number.
///
/// This can encode innovations that represent a new link forming, or a new node being added.  In each case, two
/// nodes fully specify the innovation and where it must have occurred (between them).
pub const Innovation = struct {
    /// The Id of NNode inputting into the Link where the innovation occurred.
    in_node_id: i64,
    /// The Id of NNode outputting from the Link where the innovation occurred.
    out_node_id: i64,
    /// The number assigned to the innovation.
    innovation_num: i64,
    /// If new node innovation, there are 2 innovations (Links) added for the new node.
    innnovation_num2: i64 = undefined,

    /// If Link is added, this is its weight.
    new_weight: f64 = undefined,
    /// If link is added, this is its connected Trait index.
    new_trait_num: usize = undefined,
    /// If new NNode created, this is its Id.
    new_node_id: i64 = undefined,

    /// If new NNode created, this is the innovation number of the Gene's Link it is being inserted into.
    old_innov_num: i64 = undefined,
    /// Flag indicating whether innovation is for recurrent Link.
    is_recurrent: bool = false,
    /// The type of innovation.
    innovation_type: InnovationType,
    /// Holds reference to underlying allocator, which is used to
    /// free memory when `deinit` is called.
    allocator: std.mem.Allocator,

    /// Initializes a new Innovation for a Node.
    pub fn initForNode(allocator: std.mem.Allocator, in_node_id: i64, out_node_id: i64, innovation_num_1: i64, innovation_num_2: i64, new_node_id: i64, old_innovation_num: i64) !*Innovation {
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

    /// Initializes a new Innovation for a Link.
    pub fn initForLink(allocator: std.mem.Allocator, in_node_id: i64, out_node_id: i64, innovation_num: i64, weight: f64, trait_id: usize) !*Innovation {
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

    /// Initializes a new Innovation for a Recurrent Link.
    pub fn initForRecurrentLink(allocator: std.mem.Allocator, in_node_id: i64, out_node_id: i64, innovation_num: i64, weight: f64, trait_id: usize, recur: bool) !*Innovation {
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

    /// Frees all associated memory.
    pub fn deinit(self: *Innovation) void {
        self.allocator.destroy(self);
    }
};
