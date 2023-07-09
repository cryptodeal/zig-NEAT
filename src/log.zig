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

log_level: std.log.Level = std.log.Level.err,

pub fn debug(self: *NeatLogger, comptime msg: []const u8, args: anytype, src: ?std.builtin.SourceLocation) void {
    if (@intFromEnum(self.log_level) >= @intFromEnum(std.log.Level.debug)) {
        if (src != null) {
            std.log.debug("({s}:{d}) " ++ msg, .{ src.?.file, src.?.line } ++ args);
        } else {
            std.log.debug(msg, args);
        }
    }
}

pub fn err(self: *NeatLogger, comptime msg: []const u8, args: anytype, src: ?std.builtin.SourceLocation) void {
    if (@intFromEnum(self.log_level) >= @intFromEnum(std.log.Level.err)) {
        if (src != null) {
            std.log.err("({s}:{d}) " ++ msg, .{ src.?.file, src.?.line } ++ args);
        } else {
            std.log.err(msg, args);
        }
    }
}

pub fn info(self: *NeatLogger, comptime msg: []const u8, args: anytype, src: ?std.builtin.SourceLocation) void {
    if (@intFromEnum(self.log_level) >= @intFromEnum(std.log.Level.info)) {
        if (src != null) {
            std.log.info("({s}:{d}) " ++ msg, .{ src.?.file, src.?.line } ++ args);
        } else {
            std.log.info(msg, args);
        }
    }
}

pub fn warn(self: *NeatLogger, comptime msg: []const u8, args: anytype, src: ?std.builtin.SourceLocation) void {
    if (@intFromEnum(self.log_level) >= @intFromEnum(std.log.Level.warn)) {
        if (src != null) {
            std.log.warn("({s}:{d}) " ++ msg, .{ src.?.file, src.?.line } ++ args);
        } else {
            std.log.warn(msg, args);
        }
    }
}

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
