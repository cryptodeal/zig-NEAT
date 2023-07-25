const std = @import("std");
const env = @import("environment.zig");

const Point = env.Point;

/// Record holding info about individual maze agent performance at the end of simulation
pub const AgentRecord = struct {
    /// id of the Agent
    agent_id: usize = undefined,
    /// Agent's X position at the end of simulation
    x: f64 = undefined,
    /// Agent's Y position at the end of simulation
    y: f64 = undefined,
    /// fitness value of the Agent
    fitness: f64 = undefined,
    /// flag indicating whether Agent reached maze exit
    got_exit: bool = false,
    /// the population generation when agent data was collected
    generation: usize = undefined,
    /// the associated novelty value
    novelty: f64 = undefined,

    /// The ID of species to whom individual belongs
    species_id: usize = undefined,
    /// The age of species to whom individual belongs at time of recording
    species_age: usize = undefined,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*AgentRecord {
        var self = try allocator.create(AgentRecord);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *AgentRecord) void {
        self.allocator.destroy(self);
    }
};

fn readIntLittleAny(stream: anytype, comptime T: type) !T {
    const BiggerInt = std.meta.Int(@typeInfo(T).Int.signedness, 8 * @as(usize, ((@bitSizeOf(T) + 7)) / 8));
    return @as(T, @truncate(try stream.readIntLittle(BiggerInt)));
}

/// Record storage for the maze Agent
pub const RecordStore = struct {
    /// list of the Agent's records
    records: std.ArrayList(*AgentRecord),
    /// list of the solver Agent's path points
    solver_path_points: std.ArrayList(*Point),

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*RecordStore {
        const self = try allocator.create(RecordStore);
        self.* = .{
            .allocator = allocator,
            .records = std.ArrayList(*AgentRecord).init(allocator),
            .solver_path_points = std.ArrayList(*Point).init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *RecordStore) void {
        for (self.records.items) |item| {
            item.deinit();
        }
        self.records.deinit();
        for (self.solver_path_points.items) |item| {
            item.deinit();
        }
        self.solver_path_points.deinit();
        self.allocator.destroy(self);
    }

    pub fn write(self: *RecordStore, writer: anytype) !void {
        // write records
        try writer.writeIntLittle(u64, self.records.items.len);
        for (self.records.items) |item| {
            try writer.writeIntLittle(u64, item.agent_id);
            try writer.writeIntLittle(u64, @as(u64, @bitCast(item.x)));
            try writer.writeIntLittle(u64, @as(u64, @bitCast(item.y)));
            try writer.writeIntLittle(u64, @as(u64, @bitCast(item.fitness)));
            try writer.writeByte(@intFromBool(item.got_exit));
            try writer.writeIntLittle(u64, item.generation);
            try writer.writeIntLittle(u64, @as(u64, @bitCast(item.novelty)));
            try writer.writeIntLittle(u64, item.species_id);
            try writer.writeIntLittle(u64, item.species_age);
        }
        // write solver path points
        try writer.writeIntLittle(u64, self.solver_path_points.items.len);
        for (self.solver_path_points.items) |item| {
            try writer.writeIntLittle(u64, @as(u64, @bitCast(item.x)));
            try writer.writeIntLittle(u64, @as(u64, @bitCast(item.y)));
        }
    }

    pub fn read(allocator: std.mem.Allocator, reader: anytype) !*RecordStore {
        var new_store = try allocator.create(RecordStore);

        // read AgentRecords into ArrayList
        var records_len = std.math.cast(usize, try reader.readIntLittle(u64)) orelse return error.UnexpectedData;
        var records = try std.ArrayList(*AgentRecord).initCapacity(allocator, records_len);
        var i: usize = 0;
        while (i < records_len) : (i += 1) {
            var record = try AgentRecord.init(allocator);
            record.agent_id = std.math.cast(usize, try reader.readIntLittle(u64)) orelse return error.UnexpectedData;
            record.x = @as(f64, @bitCast(try reader.readIntLittle(u64)));
            record.y = @as(f64, @bitCast(try reader.readIntLittle(u64)));
            record.fitness = @as(f64, @bitCast(try reader.readIntLittle(u64)));
            record.got_exit = (try reader.readByte()) != 0;
            record.generation = std.math.cast(usize, try reader.readIntLittle(u64)) orelse return error.UnexpectedData;
            record.novelty = @as(f64, @bitCast(try reader.readIntLittle(u64)));
            record.species_id = std.math.cast(usize, try reader.readIntLittle(u64)) orelse return error.UnexpectedData;
            record.species_age = std.math.cast(usize, try reader.readIntLittle(u64)) orelse return error.UnexpectedData;
            records.appendAssumeCapacity(record);
        }
        var points_len = std.math.cast(usize, try reader.readIntLittle(u64)) orelse return error.UnexpectedData;
        var solver_path_points = try std.ArrayList(*Point).initCapacity(allocator, points_len);
        i = 0;
        while (i < points_len) : (i += 1) {
            var x = @as(f64, @bitCast(try reader.readIntLittle(u64)));
            var y = @as(f64, @bitCast(try reader.readIntLittle(u64)));
            var point = try Point.init_coords(allocator, x, y);
            solver_path_points.appendAssumeCapacity(point);
        }
        new_store.* = .{
            .allocator = allocator,
            .records = records,
            .solver_path_points = solver_path_points,
        };
        return new_store;
    }
};

// TODO: finish implementation and verify passes tests
test "RecordStore write" {
    var allocator = std.testing.allocator;
    var rs = try RecordStore.init(allocator);
    defer rs.deinit();

    var r1 = try AgentRecord.init(allocator);
    r1.agent_id = 0;
    r1.x = 1;
    r1.y = 2;
    r1.fitness = 4;
    r1.generation = 1;
    r1.novelty = 0;
    r1.species_id = 1;
    r1.species_age = 1;
    try rs.records.append(r1);

    var r2 = try AgentRecord.init(allocator);
    r2.agent_id = 1;
    r2.x = 10;
    r2.y = 20;
    r2.fitness = 40;
    r2.generation = 1;
    r2.novelty = 0;
    r2.species_id = 1;
    r2.species_age = 1;
    try rs.records.append(r2);

    var r3 = try AgentRecord.init(allocator);
    r3.agent_id = 2;
    r3.x = 11;
    r3.y = 21;
    r3.fitness = 41;
    r3.generation = 1;
    r3.novelty = 0;
    r3.species_id = 1;
    r3.species_age = 1;
    try rs.records.append(r3);

    var r4 = try AgentRecord.init(allocator);
    r4.agent_id = 3;
    r4.x = 12;
    r4.y = 22;
    r4.fitness = 42;
    r4.generation = 1;
    r4.novelty = 0;
    r4.species_id = 1;
    r4.species_age = 1;
    try rs.records.append(r4);

    var pt1 = try Point.init_coords(allocator, 0, 1);
    try rs.solver_path_points.append(pt1);

    var pt2 = try Point.init_coords(allocator, 2, 3);
    try rs.solver_path_points.append(pt2);

    var pt3 = try Point.init_coords(allocator, 4, 5);
    try rs.solver_path_points.append(pt3);

    var og_bytes = std.ArrayList(u8).init(allocator);
    defer og_bytes.deinit();

    // write to arraylist
    try rs.write(og_bytes.writer());

    // init RecordStore from binary data stream
    var stream = std.io.fixedBufferStream(og_bytes.items);
    var new_res = try RecordStore.read(allocator, stream.reader());
    defer new_res.deinit();

    // check that the read values are the same as the written values
    var new_bytes = std.ArrayList(u8).init(allocator);
    defer new_bytes.deinit();
    try new_res.write(new_bytes.writer());

    try std.testing.expectEqualSlices(u8, og_bytes.items, new_bytes.items);
}
