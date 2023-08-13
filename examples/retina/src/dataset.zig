const std = @import("std");
const env = @import("environment.zig");

const VisualObject = env.VisualObject;

pub fn createRetinaDataset(allocator: std.mem.Allocator) ![]*VisualObject {
    var objs = std.ArrayList(*VisualObject).init(allocator);
    // set left side objects
    try objs.append(try VisualObject.init(allocator, .BothSides, ". .\n. ."));
    try objs.append(try VisualObject.init(allocator, .BothSides, ". .\n. o"));
    try objs.append(try VisualObject.init(allocator, .LeftSide, ". o\n. o"));
    try objs.append(try VisualObject.init(allocator, .BothSides, ". o\n. ."));
    try objs.append(try VisualObject.init(allocator, .LeftSide, ". o\no o"));
    try objs.append(try VisualObject.init(allocator, .BothSides, ". .\no ."));
    try objs.append(try VisualObject.init(allocator, .LeftSide, "o o\n. o"));
    try objs.append(try VisualObject.init(allocator, .BothSides, "o .\n. ."));

    // set right side objects
    try objs.append(try VisualObject.init(allocator, .BothSides, ". .\n. ."));
    try objs.append(try VisualObject.init(allocator, .BothSides, "o .\n. ."));
    try objs.append(try VisualObject.init(allocator, .RightSide, "o .\no ."));
    try objs.append(try VisualObject.init(allocator, .BothSides, ". .\no ."));
    try objs.append(try VisualObject.init(allocator, .RightSide, "o o\no ."));
    try objs.append(try VisualObject.init(allocator, .BothSides, ". o\n. ."));
    try objs.append(try VisualObject.init(allocator, .RightSide, "o .\no o"));
    try objs.append(try VisualObject.init(allocator, .BothSides, ". .\n. o"));

    return objs.toOwnedSlice();
}

test "Create Retina DataSet" {
    const allocator = std.testing.allocator;
    var dataset = try createRetinaDataset(allocator);
    defer allocator.free(dataset);

    try std.testing.expect(dataset.len == 16);
    for (dataset) |vo| {
        defer vo.deinit();
        try std.testing.expect(vo.data.len == 4);
    }
}
