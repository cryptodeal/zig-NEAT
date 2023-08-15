//! Wraps `std.log` to provide granular control of logging output.

const std = @import("std");

pub const NeatLogger = @This();

pub fn log_fn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = scope;
    const prefix = "[" ++ comptime level.asText() ++ "] ";
    std.debug.getStderrMutex().lock();
    defer std.debug.getStderrMutex().unlock();
    const stderr = std.io.getStdErr().writer();
    nosuspend stderr.print(prefix ++ format ++ "\n", args) catch return;
}

/// Specifies logger output level.
log_level: std.log.Level = std.log.Level.err,

/// output messages with `std.log.Level.debug` and up
pub fn debug(self: *NeatLogger, comptime msg: []const u8, args: anytype, src: ?std.builtin.SourceLocation) void {
    if (@intFromEnum(self.log_level) >= @intFromEnum(std.log.Level.debug)) {
        if (src != null) {
            std.log.debug("({s}:{d}) " ++ msg, .{ src.?.file, src.?.line } ++ args);
        } else {
            std.log.debug(msg, args);
        }
    }
}

/// output messages with `std.log.Level.err` and up
pub fn err(self: *NeatLogger, comptime msg: []const u8, args: anytype, src: ?std.builtin.SourceLocation) void {
    if (@intFromEnum(self.log_level) >= @intFromEnum(std.log.Level.err)) {
        if (src != null) {
            std.log.err("({s}:{d}) " ++ msg, .{ src.?.file, src.?.line } ++ args);
        } else {
            std.log.err(msg, args);
        }
    }
}

/// output messages with `std.log.Level.info` and up
pub fn info(self: *NeatLogger, comptime msg: []const u8, args: anytype, src: ?std.builtin.SourceLocation) void {
    if (@intFromEnum(self.log_level) >= @intFromEnum(std.log.Level.info)) {
        if (src != null) {
            std.log.info("({s}:{d}) " ++ msg, .{ src.?.file, src.?.line } ++ args);
        } else {
            std.log.info(msg, args);
        }
    }
}

/// output messages with `std.log.Level.warn` and up
pub fn warn(self: *NeatLogger, comptime msg: []const u8, args: anytype, src: ?std.builtin.SourceLocation) void {
    if (@intFromEnum(self.log_level) >= @intFromEnum(std.log.Level.warn)) {
        if (src != null) {
            std.log.warn("({s}:{d}) " ++ msg, .{ src.?.file, src.?.line } ++ args);
        } else {
            std.log.warn(msg, args);
        }
    }
}

/// Initializes NeatLogger (and set log level).
pub fn init(self: *NeatLogger, level: []const u8) !void {
    if (std.mem.eql(u8, level, std.log.Level.err.asText())) {
        self.log_level = std.log.Level.err;
    } else if (std.mem.eql(u8, level, std.log.Level.warn.asText())) {
        self.log_level = std.log.Level.warn;
    } else if (std.mem.eql(u8, level, std.log.Level.info.asText())) {
        self.log_level = std.log.Level.info;
    } else if (std.mem.eql(u8, level, std.log.Level.debug.asText())) {
        self.log_level = std.log.Level.debug;
    } else {
        return error.NeatLoggerInvalidLogLevel;
    }
}
