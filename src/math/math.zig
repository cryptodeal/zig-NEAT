const std = @import("std");

pub fn rand_sign(comptime T: type) T {
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.os.getrandom(std.mem.asBytes(&seed)) catch {
            seed = 902999832;
        };
        break :blk seed;
    });
    const rand = prng.random();
    const v = rand.int(i64);

    if (@rem(v, 2) == 0) {
        return @as(T, @intCast(-1));
    } else {
        return @as(T, @intCast(1));
    }
}

pub fn single_roulette_throw(probabilities: []f64) i64 {
    var total: f64 = 0.0;

    for (probabilities) |v| {
        total += v;
    }

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.os.getrandom(std.mem.asBytes(&seed)) catch {
            seed = 902999832;
        };
        break :blk seed;
    });
    const rand = prng.random();

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

    while (i < runs) : (i += 1) {
        const idx = @as(usize, @intCast(single_roulette_throw(&probabilities)));
        if (idx < 0 or idx >= probabilities.len) {
            try std.testing.expect(false);
            std.debug.print("\ninvalid segment index: {d} at {d}\n", .{ idx, i });
        }
        // increment histogram to check distribution quality
        hist[idx] += 1;
    }
}
