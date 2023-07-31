const std = @import("std");

pub fn get_writable_file(path: []const u8) !std.fs.File {
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

pub fn read_file(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
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
