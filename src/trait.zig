const std = @import("std");
const math = @import("math/math.zig");

pub const NumTraitParams: usize = 8;

pub const Trait = struct {
    // Trait id
    id: ?i64,
    // learned Trait parameters
    params: []f64,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, length: usize) !*Trait {
        var t = try allocator.create(Trait);
        t.* = .{
            .id = undefined,
            .allocator = allocator,
            .params = try allocator.alloc(f64, length),
        };
        for (t.params) |*x| x.* = 0;

        return t;
    }

    pub fn new_trait_copy(allocator: std.mem.Allocator, trait: *Trait) !*Trait {
        var nt = try Trait.init(allocator, trait.params.len);
        nt.id = trait.id;
        @memcpy(nt.params, trait.params);
        return nt;
    }

    pub fn new_trait_avg(allocator: std.mem.Allocator, trait_1: *Trait, trait_2: *Trait) !*Trait {
        if (trait_1.params.len != trait_2.params.len) {
            std.debug.print("traits parameters number mismatch; {d} != {d}\n", .{ trait_1.params.len, trait_2.params.len });
            return error.TraitsParametersCountMismatch;
        }
        var nt = try Trait.init(allocator, trait_1.params.len);
        nt.id = trait_1.id;
        for (trait_1.params, 0..) |p, i| {
            nt.params[i] = (p + trait_2.params[i]) / 2.0;
        }
        return nt;
    }

    pub fn read_from_file(allocator: std.mem.Allocator, data: []const u8) !*Trait {
        var self = try Trait.init(allocator, NumTraitParams);
        errdefer self.deinit();
        var split = std.mem.split(u8, data, " ");
        self.id = try std.fmt.parseInt(i64, split.first(), 10);
        var count: usize = 0;
        while (count < NumTraitParams) : (count += 1) {
            var val = split.next();
            if (val == null) return error.MalformedTraitInGenomeFile;
            self.params[count] = try std.fmt.parseFloat(f64, val.?);
        }
        return self;
    }

    pub fn deinit(self: *Trait) void {
        self.allocator.free(self.params);
        self.allocator.destroy(self);
    }

    pub fn is_equal(self: *Trait, t: *Trait) bool {
        if (self.params.len != t.params.len or self.id.? != t.id.?) {
            return false;
        }
        return std.mem.eql(f64, self.params, t.params);
    }

    pub fn mutate(self: *Trait, mutation_power: f64, param_mutate_prob: f64) void {
        var prng = std.rand.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            std.os.getrandom(std.mem.asBytes(&seed)) catch {
                seed = 902999832;
            };
            break :blk seed;
        });
        const rand = prng.random();
        for (self.params) |*p| {
            if (rand.float(f64) > param_mutate_prob) {
                p.* += @as(f64, @floatFromInt(math.rand_sign(i32))) * rand.float(f64) * mutation_power;
                if (p.* < 0) {
                    p.* = 0;
                }
            }
        }
    }

    pub fn format(value: Trait, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Trait #{any} (", .{value.id});
        for (value.params) |p| {
            try writer.print(" {d}", .{p});
        }
        return writer.print(" )", .{});
    }
};

test "new trait average" {
    const allocator = std.testing.allocator;
    var t1 = try Trait.init(allocator, NumTraitParams);
    defer t1.deinit();
    for (t1.params, 0..) |_, i| {
        t1.params[i] = @as(f64, @floatFromInt(i + 1));
    }
    var t2 = try Trait.init(allocator, NumTraitParams);
    defer t2.deinit();
    for (t2.params, 0..) |_, i| {
        t2.params[i] = @as(f64, @floatFromInt(i + 2));
    }

    var new_trait = try Trait.new_trait_avg(allocator, t1, t2);
    defer new_trait.deinit();
    for (new_trait.params, 0..) |p, i| {
        const expected = (t1.params[i] + t2.params[i]) / 2.0;
        try std.testing.expect(expected == p);
    }
}

test "new trait copy" {
    const allocator = std.testing.allocator;
    var t1 = try Trait.init(allocator, NumTraitParams);
    defer t1.deinit();
    t1.id = 1;
    for (t1.params, 0..) |_, i| {
        t1.params[i] = @as(f64, @floatFromInt(i + 1));
    }
    var t2 = try Trait.new_trait_copy(allocator, t1);
    defer t2.deinit();
    try std.testing.expect(t1.id == t2.id);
    for (t2.params, 0..) |p, i| {
        try std.testing.expect(t1.params[i] == p);
    }
}

test {
    std.testing.refAllDecls(@This());
}
