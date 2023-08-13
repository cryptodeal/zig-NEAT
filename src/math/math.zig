const std = @import("std");

pub const NodeActivationType = @import("activations.zig").NodeActivationType;

pub fn randSign(comptime T: type, rand: std.rand.Random) T {
    const v = rand.int(i64);
    if (@rem(v, 2) == 0) {
        return @as(T, @intCast(-1));
    } else {
        return @as(T, @intCast(1));
    }
}

pub fn singleRouletteThrow(rand: std.rand.Random, probabilities: []f64) i64 {
    var total: f64 = 0.0;

    for (probabilities) |v| {
        total += v;
    }

    // throw ball & collect result
    var throw_value = rand.float(f64) * total;

    var accumulator: f64 = 0.0;
    for (probabilities, 0..) |v, i| {
        accumulator += v;
        if (throw_value <= accumulator) {
            return @as(i64, @intCast(i));
        }
    }

    return -1;
}

test "network math tests" {
    const allocator = std.testing.allocator;
    var probabilities = [_]f64{ 0.1, 0.2, 0.4, 0.15, 0.15 };
    var hist = try allocator.alloc(f64, probabilities.len);
    defer allocator.free(hist);
    const runs: usize = 10000;
    var i: usize = 0;

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    while (i < runs) : (i += 1) {
        const idx = @as(usize, @intCast(singleRouletteThrow(rand, &probabilities)));
        if (idx < 0 or idx >= probabilities.len) {
            try std.testing.expect(false);
            std.debug.print("\ninvalid segment index: {d} at {d}\n", .{ idx, i });
        }
        // increment histogram to check distribution quality
        hist[idx] += 1;
    }
}

test {
    std.testing.refAllDecls(@This());
}
