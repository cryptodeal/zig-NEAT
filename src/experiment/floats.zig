const std = @import("std");

pub fn min(comptime T: type, x: []T) T {
    if (x.len == 0) {
        return std.math.nan(T);
    }
    return std.mem.min(T, x);
}

pub fn max(comptime T: type, x: []T) T {
    if (x.len == 0) {
        return std.math.nan(T);
    }
    return std.mem.max(T, x);
}

pub fn sum(comptime T: type, x: []T) T {
    var res: T = 0;
    for (x) |v| {
        res += v;
    }
    return res;
}

pub fn mean(comptime T: type, x: []T) T {
    if (x.len == 0) {
        return std.math.nan(T);
    }
    var total: T = 0;
    for (x) |v| {
        total += v;
    }
    return total / @as(T, @floatFromInt(x.len));
}

fn has_nan(comptime T: type, x: []T) bool {
    var contains_nan = false;
    for (x) |v| {
        if (std.math.isNan(v)) {
            contains_nan = true;
            break;
        }
    }
    return contains_nan;
}

pub fn mean_variance(allocator: std.mem.Allocator, comptime T: type, x: []T) ![]T {
    var res = try allocator.alloc(T, 2);
    if (x.len == 0) {
        for (res) |*v| v.* = std.math.nan(T);
        return res;
    }
    var ss: T = 0;
    var compensation: T = 0;
    res[0] = mean(T, x);
    for (x) |v| {
        var d: T = v - res[0];
        ss += d * d;
        compensation += d;
    }
    var unnormalized_variance: T = (ss - compensation * compensation / @as(T, @floatFromInt(x.len)));
    res[1] = unnormalized_variance / (@as(T, @floatFromInt(x.len)) - 1);
    return res;
}

pub fn median(comptime T: type, x: []T) !T {
    if (x.len == 0) {
        return std.math.nan(T);
    }
    return try quantile(T, 0.5, CumulantKind.Empirical, x, null);
}

pub fn q25(comptime T: type, x: []T) !T {
    if (x.len == 0) {
        return std.math.nan(T);
    }
    return try quantile(T, 0.25, CumulantKind.Empirical, x, null);
}

pub fn q75(comptime T: type, x: []T) !T {
    if (x.len == 0) {
        return std.math.nan(T);
    }
    return try quantile(T, 0.75, CumulantKind.Empirical, x, null);
}

pub fn variance(allocator: std.mem.Allocator, comptime T: type, x: []T) !T {
    if (x.len == 0) {
        return std.math.nan(T);
    }
    var res = try mean_variance(allocator, T, x);
    defer allocator.free(res);
    return res[1];
}

pub fn std_dev(allocator: std.mem.Allocator, comptime T: type, x: []T) !T {
    if (x.len == 0) {
        return std.math.nan(T);
    }
    var res = try mean_variance(allocator, T, x);
    defer allocator.free(res);
    return @sqrt(res[1]);
}

// util functions used internally

fn is_sorted(comptime T: type, x: []T) bool {
    var n = x.len;
    var i = n - 1;
    while (i > 0) : (i -= 1) {
        if (x[i] < x[i - 1] or (std.math.isNan(x[i]) and !std.math.isNan(x[i - 1]))) {
            return false;
        }
    }
    return true;
}

// List of supported CumulantKind values for the Quantile function.
// Constant values should match the R nomenclature. See
// https://en.wikipedia.org/wiki/Quantile#Estimating_the_quantiles_of_a_population
pub const CumulantKind = enum(u8) {
    // Empirical treats the distribution as the actual empirical distribution.
    Empirical,
    // LinInterp linearly interpolates the empirical distribution between sample values, with a flat extrapolation.
    LinInterp,
};

fn quantile(comptime T: type, p: T, c: CumulantKind, x: []T, weights: ?[]T) !T {
    if (!(p >= 0 and p <= 1)) {
        std.debug.print("percentile out of bounds", .{});
        return error.PercentileOutOfBounds;
    }
    if (weights != null and x.len != weights.?.len) {
        std.debug.print("slice length mismatch", .{});
        return error.SliceLengthMismatch;
    }
    if (x.len == 0) {
        std.debug.print("zero length slice", .{});
        return error.ZeroLengthSlice;
    }
    if (has_nan(T, x)) {
        // This is needed because the algorithm breaks otherwise.
        return std.math.nan(T);
    }
    if (!is_sorted(T, x)) {
        std.debug.print("x data are not sorted", .{});
        return error.DataNotSorted;
    }
    var sum_weights: T = undefined;
    if (weights == null) {
        sum_weights = @as(T, @floatFromInt(x.len));
    } else {
        sum_weights = sum(T, weights.?);
    }
    return switch (c) {
        CumulantKind.Empirical => empirical_quantile(T, p, x, weights, sum_weights),
        CumulantKind.LinInterp => lin_interp_quantile(T, p, x, weights, sum_weights),
    };
}

fn empirical_quantile(comptime T: type, p: T, x: []T, weights: ?[]T, sum_weights: T) T {
    var cumsum: T = undefined;
    var fidx = p * sum_weights;
    for (x, 0..) |_, i| {
        if (weights == null) {
            cumsum += 1;
        } else {
            cumsum += weights.?[i];
        }
        if (cumsum >= fidx) {
            return x[i];
        }
    }
    unreachable;
}

fn lin_interp_quantile(comptime T: type, p: T, x: []T, weights: ?[]T, sum_weights: T) T {
    var cumsum: T = undefined;
    var fidx = p * sum_weights;
    for (x, 0..) |_, i| {
        if (weights == null) {
            cumsum += 1;
        } else {
            cumsum += weights.?[i];
        }
        if (cumsum >= fidx) {
            if (i == 0) {
                return x[0];
            }
            var t = cumsum - fidx;
            if (weights != null) {
                t /= weights.?[i];
            }
            return t * x[i - 1] + (1 - t) * x[i];
        }
    }
    unreachable;
}
