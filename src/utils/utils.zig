const std = @import("std");

/// Given relative path from CWD, returns a writable file handle.
/// Caller is responsible for closing the file handle.
pub fn getWritableFile(path: []const u8) !std.fs.File {
    const dir_path = std.fs.path.dirname(path);
    const file_name = std.fs.path.basename(path);
    var file_dir: std.fs.Dir = undefined;
    defer file_dir.close();
    if (dir_path != null) {
        file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
    } else {
        file_dir = std.fs.cwd();
    }
    return file_dir.createFile(file_name, .{});
}

/// Given relative path from CWD, reads entire file into buffer. Caller is responsible for
/// freeing allocated memory of returned buffer.
pub fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const dir_path = std.fs.path.dirname(path);
    const file_name = std.fs.path.basename(path);
    var file_dir: std.fs.Dir = undefined;
    defer file_dir.close();
    if (dir_path != null) {
        file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
    } else {
        file_dir = std.fs.cwd();
    }
    var file = try file_dir.openFile(file_name, .{});
    defer file.close();
    const file_size = (try file.stat()).size;
    var buf = try allocator.alloc(u8, file_size);
    try file.reader().readNoEof(buf);
    return buf;
}
