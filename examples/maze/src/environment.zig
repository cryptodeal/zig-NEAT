const std = @import("std");

/// The maximal allowed speed for maze agent
pub const max_agent_speed: f64 = 3;

/// The simple Point struct
pub const Point = struct {
    x: f64 = undefined,
    y: f64 = undefined,
    allocator: std.mem.Allocator = undefined,

    pub fn init(allocator: std.mem.Allocator) !*Point {
        var self = try allocator.create(Point);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn init_coords(allocator: std.mem.Allocator, x: f64, y: f64) !*Point {
        var self = try Point.init(allocator);
        self.x = x;
        self.y = y;
        return self;
    }

    pub fn clone(self: *Point, allocator: std.mem.Allocator) !*Point {
        return Point.init_coords(allocator, self.x, self.y);
    }

    pub fn read(allocator: std.mem.Allocator, data: []const u8) !*Point {
        var self = try Point.init(allocator);
        var split = std.mem.split(u8, data, " ");
        self.x = try std.fmt.parseFloat(f64, split.first());
        self.y = try std.fmt.parseFloat(f64, split.rest());
        return self;
    }

    pub fn tmp_copy(self: *Point) Point {
        return Point{ .x = self.x, .y = self.y };
    }

    pub fn deinit(self: *Point) void {
        self.allocator.destroy(self);
    }

    /// Used to determine angle in degrees of vector defined by (0,0)->this Point.
    // The angle is from 0 to 360 degrees counter-clockwise.
    pub fn angle(self: *Point) f64 {
        var ang = std.math.atan2(f64, self.y, self.x) / @as(f64, std.math.pi) * 180;
        if (ang < 0) {
            // lower quadrants (3 and 4)
            return ang + 360;
        }
        return ang;
    }

    /// rotates this point around another point with given angle in degrees
    pub fn rotate(self: *Point, ang: f64, point: *Point) void {
        var rad = ang / 180 * @as(f64, std.math.pi);
        self.x -= point.x;
        self.y -= point.y;

        var ox = self.x;
        var oy = self.y;
        self.x = @cos(rad) * ox - @sin(rad) * oy;
        self.y = @sin(rad) * ox + @cos(rad) * oy;

        self.x += point.x;
        self.y += point.y;
    }

    /// finds the distance between this point and another point
    pub fn distance(self: *Point, point: *Point) f64 {
        var dx = point.x - self.x;
        var dy = point.y - self.y;
        return @sqrt(dx * dx + dy * dy);
    }
};

pub const Line = struct {
    a: *Point,
    b: *Point,
    allocator: std.mem.Allocator = undefined,

    pub fn init(allocator: std.mem.Allocator, a: *Point, b: *Point) !*Line {
        var self = try allocator.create(Line);
        self.* = .{
            .allocator = allocator,
            .a = a,
            .b = b,
        };
        return self;
    }

    pub fn clone(self: *Line, allocator: std.mem.Allocator) !*Line {
        return Line.init(allocator, try self.a.clone(allocator), try self.b.clone(allocator));
    }

    pub fn read(allocator: std.mem.Allocator, data: []const u8) !*Line {
        var split = std.mem.split(u8, data, " ");
        var a = try Point.init(allocator);
        var b = try Point.init(allocator);
        comptime var i: usize = 0;
        inline while (i < 4) : (i += 1) {
            switch (i) {
                0 => a.x = try std.fmt.parseFloat(f64, split.next().?),
                1 => a.y = try std.fmt.parseFloat(f64, split.next().?),
                2 => b.x = try std.fmt.parseFloat(f64, split.next().?),
                3 => b.y = try std.fmt.parseFloat(f64, split.next().?),
                else => unreachable,
            }
        }

        return Line.init(allocator, a, b);
    }

    pub fn deinit(self: *Line) void {
        self.a.deinit();
        self.b.deinit();
        self.allocator.destroy(self);
    }

    /// finds the midpoint of this line
    pub fn midpoint(self: *Line) Point {
        return Point{
            .x = (self.a.x + self.b.x) / 2,
            .y = (self.a.y + self.b.y) / 2,
        };
    }

    /// calculates point of intersection between two line segments if it exists
    pub fn intersection(self: *Line, line: *Line) ?Point {
        var pt: ?Point = null;
        var a = self.a;
        var b = self.b;
        var c = line.a;
        var d = line.b;

        var r_top = (a.y - c.y) * (d.x - c.x) - (a.x - c.x) * (d.y - c.y);
        var r_bot = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x);

        var s_top = (a.y - c.y) * (b.x - a.x) - (a.x - c.x) * (b.y - a.y);
        var s_bot = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x);

        if (r_bot == 0 or s_bot == 0) {
            // lines are parallel
            return pt;
        }

        var r = r_top / r_bot;
        var s = s_top / s_bot;
        if (r > 0 and r < 1 and s > 0 and s < 1) {
            pt = Point{ .x = a.x + r * (b.x - a.x), .y = a.y + r * (b.y - a.y) };
        }
        return pt;
    }

    /// find distance between Line segment and the Point
    pub fn distance(self: *Line, p: *Point) f64 {
        var utop = (p.x - self.a.x) * (self.b.x - self.a.x) + (p.y - self.a.y) * (self.b.y - self.a.y);
        var ubot = self.a.distance(self.b);
        ubot *= ubot;
        if (ubot == 0) {
            return 0;
        }

        var u = utop / ubot;
        if (u < 0 or u > 1) {
            var d1 = self.a.distance(p);
            var d2 = self.b.distance(p);
            if (d1 < d2) {
                return d1;
            }
            return d2;
        }
        var pt = Point{};
        pt.x = self.a.x + u * (self.b.x - self.a.x);
        pt.y = self.a.y + u * (self.b.y - self.a.y);
        return pt.distance(p);
    }

    /// calculates the length of this line segment
    pub fn length(self: *Line) f64 {
        return self.a.distance(self.b);
    }
};

const init_range_finder_angles = [_]f64{ -90, -45, 0, 45, 90, -180 };
const init_radar_angles1 = [_]f64{ 315, 45, 135, 225 };
const init_radar_angles2 = [_]f64{ 405, 135, 225, 315 };

/// Agent represents the maze navigating Agent
pub const Agent = struct {
    /// the current position of the Agent
    location: *Point = undefined,
    /// heading direction in degrees of the Agent
    heading: f64 = 0,
    /// the speed of the Agent
    speed: f64 = 0,
    /// the angular velocity of the Agent
    angular_velocity: f64 = 0,
    /// radius of the Agent's body
    radius: f64 = 8,
    /// the maximal range of range finder sensors
    range_finder_range: f64 = 100,

    /// the angles of range finder sensors
    range_finder_angles: []f64,
    /// the beginning angles for radar sensors
    radar_angles1: []f64,
    /// the ending angles for radar sensors
    radar_angles2: []f64,

    /// stores the radar outputs
    radar: []f64,
    /// stores range finder outputs
    range_finders: []f64,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*Agent {
        var self = try allocator.create(Agent);
        self.* = .{
            .allocator = allocator,
            .range_finder_angles = try allocator.alloc(f64, 6),
            .radar_angles1 = try allocator.alloc(f64, 4),
            .radar_angles2 = try allocator.alloc(f64, 4),
            .range_finders = try allocator.alloc(f64, 6),
            .radar = try allocator.alloc(f64, 4),
        };
        @memcpy(self.range_finder_angles, &init_range_finder_angles);
        @memcpy(self.radar_angles1, &init_radar_angles1);
        @memcpy(self.radar_angles2, &init_radar_angles2);
        return self;
    }

    pub fn deinit(self: *Agent) void {
        self.location.deinit();
        self.allocator.free(self.range_finder_angles);
        self.allocator.free(self.radar_angles1);
        self.allocator.free(self.radar_angles2);
        self.allocator.free(self.range_finders);
        self.allocator.free(self.radar);
        self.allocator.destroy(self);
    }

    pub fn clone(self: *Agent, allocator: std.mem.Allocator) !*Agent {
        var new_agent = try allocator.create(Agent);
        new_agent.* = .{
            .allocator = allocator,
            .location = try self.location.clone(allocator),
            .heading = self.heading,
            .speed = self.speed,
            .angular_velocity = self.angular_velocity,
            .radius = self.radius,
            .range_finder_range = self.range_finder_range,
            .range_finder_angles = try allocator.alloc(f64, self.range_finder_angles.len),
            .radar_angles1 = try allocator.alloc(f64, self.radar_angles1.len),
            .radar_angles2 = try allocator.alloc(f64, self.radar_angles2.len),
            .range_finders = try allocator.alloc(f64, self.range_finders.len),
            .radar = try allocator.alloc(f64, self.radar.len),
        };
        @memcpy(new_agent.range_finder_angles, self.range_finder_angles);
        @memcpy(new_agent.radar_angles1, self.radar_angles1);
        @memcpy(new_agent.radar_angles2, self.radar_angles2);
        @memcpy(new_agent.range_finders, self.range_finders);
        @memcpy(new_agent.radar, self.radar);
        return new_agent;
    }
};

// Struct defining the maze Environment
pub const Environment = struct {
    /// the Agent navigating the maze Environment
    hero: *Agent,
    /// the list of line segments the maze is comprised of
    lines: []*Line = undefined,
    /// the point marking the exit of the maze (the goal)
    maze_exit: *Point = undefined,

    /// flag indicating whether the exit was found
    exit_found: bool = false,

    /// the number of time steps to be executed during maze solving simulation
    time_steps: usize = undefined,
    /// the sample step size to determine when to collect subsequent samples during simulation
    sample_size: usize = undefined,

    /// the range around maze exit point to test if agent coordinates is within to be considered as solved successfully (5.0 is good enough)
    exit_found_range: f64 = undefined,

    /// Agent's initial distance from the exit
    initial_distance: f64 = undefined,

    allocator: std.mem.Allocator,

    pub fn clone(self: *Environment, allocator: std.mem.Allocator) !*Environment {
        var new_env = try allocator.create(Environment);
        new_env.* = .{
            .allocator = allocator,
            .hero = try self.hero.clone(allocator),
            .lines = try allocator.alloc(*Line, self.lines.len),
            .maze_exit = try self.maze_exit.clone(allocator),
            .exit_found = self.exit_found,
            .time_steps = self.time_steps,
            .sample_size = self.sample_size,
            .exit_found_range = self.exit_found_range,
            .initial_distance = self.initial_distance,
        };

        for (self.lines, 0..) |line, i| {
            new_env.lines[i] = try line.clone(allocator);
        }

        return new_env;
    }

    pub fn read_from_file(allocator: std.mem.Allocator, path: []const u8) !*Environment {
        // read the file
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
        defer allocator.free(buf);
        try file.reader().readNoEof(buf);

        // init environment
        var self = try allocator.create(Environment);
        errdefer self.deinit();
        self.* = .{
            .allocator = allocator,
            .hero = try Agent.init(allocator),
        };
        var lines = std.ArrayList(*Line).init(allocator);

        // parse file
        var new_line_iterator = std.mem.split(u8, buf, "\n");
        var idx: usize = 0;
        var num_lines: usize = 0;
        while (new_line_iterator.next()) |raw_line| {
            var line = std.mem.trim(u8, raw_line, &std.ascii.whitespace);
            if (line.len == 0 or std.mem.startsWith(u8, line, "#")) continue; // skip empty lines and comments
            switch (idx) {
                // read in how many line segments
                0 => num_lines = try std.fmt.parseInt(usize, line, 10),
                // read initial agent's location
                1 => self.hero.location = try Point.read(allocator, line),
                // read initial heading
                2 => self.hero.heading = try std.fmt.parseFloat(f64, line),
                // read the maze exit location
                3 => self.maze_exit = try Point.read(allocator, line),
                // read maze line segments
                else => {
                    var maze_line = try Line.read(allocator, line);
                    try lines.append(maze_line);
                },
            }
            idx += 1;
        }
        self.lines = try lines.toOwnedSlice();

        if (num_lines != self.lines.len) {
            std.debug.print("Expected: {d} maze lines, but was read only: {d}", .{ num_lines, self.lines.len });
            return error.MalformedMazeFile;
        }

        // update sensors
        try self.update_range_finders();
        self.update_radar();

        // find initial distance
        self.initial_distance = self.agent_distance_to_exit();

        return self;
    }

    pub fn deinit(self: *Environment) void {
        self.hero.deinit();
        self.maze_exit.deinit();
        for (self.lines) |line| {
            line.deinit();
        }
        self.allocator.free(self.lines);
        self.allocator.destroy(self);
    }

    /// create neural net inputs from maze agent sensors
    pub fn get_inputs(self: *Environment, allocator: std.mem.Allocator) ![]f64 {
        var inputs_size: usize = self.hero.range_finders.len + self.hero.radar.len + 1;
        var inputs = try allocator.alloc(f64, inputs_size);
        // bias
        inputs[0] = 1;

        // range finders
        var i: usize = 0;
        while (i < self.hero.range_finders.len) : (i += 1) {
            inputs[i + 1] = self.hero.range_finders[i] / self.hero.range_finder_range;
            if (std.math.isNan(inputs[i + 1])) {
                std.debug.print("NAN in inputs from range finders\n", .{});
                return error.NanInInputs;
            }
        }

        // radar
        for (self.hero.radar, 0..) |v, j| {
            inputs[i + j] = v;
            if (std.math.isNan(inputs[i + j])) {
                std.debug.print("NAN in inputs from range finders\n", .{});
                return error.NanInInputs;
            }
        }

        return inputs;
    }

    /// transform neural net outputs into angular velocity and speed
    pub fn apply_outputs(self: *Environment, o1: f64, o2: f64) !void {
        if (std.math.isNan(o1) or std.math.isNan(o2)) {
            std.debug.print("OUTPUT is NAN\n", .{});
            return error.OutputIsNaN;
        }

        self.hero.angular_velocity += o1 - 0.5;
        self.hero.speed += o2 - 0.5;

        // constraints of speed & angular velocity
        if (self.hero.speed > max_agent_speed) {
            self.hero.speed = max_agent_speed;
        } else if (self.hero.speed < -max_agent_speed) {
            self.hero.speed = -max_agent_speed;
        }

        if (self.hero.angular_velocity > max_agent_speed) {
            self.hero.angular_velocity = max_agent_speed;
        } else if (self.hero.angular_velocity < -max_agent_speed) {
            self.hero.angular_velocity = -max_agent_speed;
        }
    }

    /// performs one time step of the simulation
    pub fn update(self: *Environment) !void {
        if (self.exit_found) return;

        // get horizontal and vertical velocity components
        var vx = @cos(self.hero.heading / 180 * @as(f64, std.math.pi)) * self.hero.speed;
        var vy = @sin(self.hero.heading / 180 * @as(f64, std.math.pi)) * self.hero.speed;

        if (std.math.isNan(vx)) {
            std.debug.print("VX is NAN\n", .{});
            return error.VxIsNaN;
        }

        if (std.math.isNan(vy)) {
            std.debug.print("VY is NAN\n", .{});
            return error.VyIsNaN;
        }

        // Update agent heading
        self.hero.heading += self.hero.angular_velocity;
        if (std.math.isNan(self.hero.angular_velocity)) {
            std.debug.print("Agent's Angular Velocity is NAN\n", .{});
            return error.AngularVelocityIsNaN;
        }

        if (self.hero.heading > 360) {
            self.hero.heading -= 360;
        }
        if (self.hero.heading < 0) {
            self.hero.heading += 360;
        }

        // find next agent's location
        var new_loc = Point{
            .x = vx + self.hero.location.x,
            .y = vy + self.hero.location.y,
        };
        if (!self.test_agent_collision(&new_loc)) {
            self.hero.location.x = new_loc.x;
            self.hero.location.y = new_loc.y;
        }
        try self.update_range_finders();
        self.update_radar();

        // check whether updated agent's position solved the maze
        self.exit_found = self.test_exit_found_by_agent();
    }

    /// tests whether agent location is within maze exit range
    pub fn test_exit_found_by_agent(self: *Environment) bool {
        if (self.exit_found) return true;

        var dist = self.agent_distance_to_exit();
        return dist < self.exit_found_range;
    }

    /// used for fitness calculations based on distance of maze Agent to the target maze exit
    pub fn agent_distance_to_exit(self: *Environment) f64 {
        return self.hero.location.distance(self.maze_exit);
    }

    pub fn update_range_finders(self: *Environment) !void {
        // iterate through each sensor and find distance to maze lines with agent's range finder sensors
        for (self.hero.range_finder_angles, 0..) |angle, i| {
            // radians...
            var rad = angle / 180 * @as(f64, std.math.pi);

            // project a point from the hero's location outwards
            var projection_point = Point{
                .x = self.hero.location.x + @cos(rad) * self.hero.range_finder_range,
                .y = self.hero.location.y + @sin(rad) * self.hero.range_finder_range,
            };

            // rotate the projection point by the hero's heading
            projection_point.rotate(self.hero.heading, self.hero.location);

            // create a line segment from the hero's location to projected
            var projection_line = Line{
                .a = self.hero.location,
                .b = &projection_point,
            };

            // set range to max by default
            var min_range = self.hero.range_finder_range;

            // now test against the environment to see if we hit anything
            for (self.lines) |line| {
                var intersection = line.intersection(&projection_line);
                if (intersection != null) {
                    // if so, then update the range to the distance
                    var found_range = intersection.?.distance(self.hero.location);

                    // we want the closest intersection
                    if (found_range < min_range) {
                        min_range = found_range;
                    }
                }
            }

            if (std.math.isNan(min_range)) {
                std.debug.print("RANGE is NAN\n", .{});
                return error.RangeIsNaN;
            }
            self.hero.range_finders[i] = min_range;
        }
    }

    /// updates radar sensors
    pub fn update_radar(self: *Environment) void {
        var target = self.maze_exit.tmp_copy();

        // rotate goal with respect to heading of agent to compensate agent's heading angle relative to zero heading angle
        target.rotate(-self.hero.heading, self.hero.location);

        // translate with respect to location of agent to compensate agent's position relative to (0,0)
        target.x -= self.hero.location.x;
        target.y -= self.hero.location.y;

        // what angle is the vector between target & agent (agent is placed into (0,0) with zero heading angle due
        // to the affine transforms above)
        var angle = target.angle();

        // fire the appropriate radar sensor
        for (self.hero.radar_angles1, 0..) |_, i| {
            self.hero.radar[i] = 0;
            if ((angle >= self.hero.radar_angles1[i] and angle < self.hero.radar_angles2[i]) or (angle + 360 >= self.hero.radar_angles1[i] and angle + 360 < self.hero.radar_angles2[i])) {
                self.hero.radar[i] = 1;
            }
        }
    }

    /// tests whether provided new location hits anything in maze
    pub fn test_agent_collision(self: *Environment, loc: *Point) bool {
        for (self.lines) |line| {
            if (line.distance(loc) < self.hero.radius) {
                return true;
            }
        }
        return false;
    }

    /// used to format Environment for printing (or capture val as string)
    pub fn format(value: Environment, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("MAZE\nHero at: {d:.1}, {d:.1}\n", .{ value.hero.location.x, value.hero.location.y });
        try writer.print("Exit at: {d:.1}, {d:.1}\n", .{ value.maze_exit.x, value.maze_exit.y });
        try writer.print("Initial distance from exit: {d}, # of simulation steps: {d}, path sampling size: {d} \n", .{ value.initial_distance, value.time_steps, value.sample_size });
        try writer.print("Lines:\n", .{});
        for (value.lines) |line| {
            try writer.print("\t[{d:.1}, {d:.1}] -> [{d:.1}, {d:.1}]\n", .{ line.a.x, line.a.y, line.b.x, line.b.y });
        }
    }
};

test "Point angle" {
    // 0°
    var p = Point{ .x = 1, .y = 0 };
    var angle = p.angle();
    try std.testing.expect(angle == 0);

    // 90°
    p.x = 0;
    p.y = 1;
    angle = p.angle();
    try std.testing.expect(angle == 90);

    // 180°
    p.x = -1;
    p.y = 0;
    angle = p.angle();
    try std.testing.expect(angle == 180);

    // 270°
    p.x = 0;
    p.y = -1;
    angle = p.angle();
    try std.testing.expect(angle == 270);

    // 45°
    p.x = 1;
    p.y = 1;
    angle = p.angle();
    try std.testing.expect(angle == 45);

    // 135°
    p.x = -1;
    p.y = 1;
    angle = p.angle();
    try std.testing.expect(angle == 135);

    // 225°
    p.x = -1;
    p.y = -1;
    angle = p.angle();
    try std.testing.expect(angle == 225);

    // 315°
    p.x = 1;
    p.y = -1;
    angle = p.angle();
    try std.testing.expect(angle == 315);
}

test "Point rotate" {
    var p = Point{ .x = 2, .y = 1 };
    var p2 = Point{ .x = 1, .y = 1 };
    p.rotate(90, &p2);
    try std.testing.expect(p.x == 1);
    try std.testing.expect(p.y == 2);

    p.rotate(180, &p2);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), p.x, 0.00000001);
    try std.testing.expect(p.y == 0);
}

test "Point distance" {
    var p = Point{ .x = 2, .y = 1 };
    var p1 = Point{ .x = 5, .y = 1 };

    var d = p.distance(&p1);
    try std.testing.expect(d == 3);

    var p2 = Point{ .x = 5, .y = 3 };
    d = p.distance(&p2);
    var expected = @sqrt(@as(f64, 13));
    try std.testing.expectApproxEqAbs(expected, d, 0.00000001);
}

test "Point read from string" {
    var allocator = std.testing.allocator;
    const str = "10 20";
    var point = try Point.read(allocator, str);
    defer point.deinit();
    try std.testing.expect(point.x == 10);
    try std.testing.expect(point.y == 20);
}

test "Line intersection" {
    var p1a = Point{ .x = 1, .y = 1 };
    var p1b = Point{ .x = 5, .y = 5 };
    var l1 = Line{ .a = &p1a, .b = &p1b };
    var p2a = Point{ .x = 1, .y = 5 };
    var p2b = Point{ .x = 5, .y = 1 };
    var l2 = Line{ .a = &p2a, .b = &p2b };

    // test intersection
    var p = l1.intersection(&l2);
    try std.testing.expect(p != null);
    try std.testing.expect(p.?.x == 3);
    try std.testing.expect(p.?.y == 3);

    // test parallel
    var p3a = Point{ .x = 2, .y = 1 };
    var p3b = Point{ .x = 6, .y = 1 };
    var l3 = Line{ .a = &p3a, .b = &p3b };
    p = l1.intersection(&l3);
    try std.testing.expect(p == null);

    // test no intersection by coordinates
    var p4a = Point{ .x = 4, .y = 4 };
    var p4b = Point{ .x = 6, .y = 1 };
    var l4 = Line{ .a = &p4a, .b = &p4b };
    p = l1.intersection(&l4);
    try std.testing.expect(p == null);
}

test "Line distance" {
    var p1a = Point{ .x = 1, .y = 1 };
    var p1b = Point{ .x = 5, .y = 1 };
    var l = Line{ .a = &p1a, .b = &p1b };
    var p = Point{ .x = 4, .y = 3 };
    var d = l.distance(&p);
    try std.testing.expect(d == 2);
}

test "Line length" {
    var p1a = Point{ .x = 1, .y = 1 };
    var p1b = Point{ .x = 5, .y = 1 };
    var l = Line{ .a = &p1a, .b = &p1b };
    var length = l.length();
    try std.testing.expect(length == 4);
}

test "Line read from string" {
    var allocator = std.testing.allocator;
    const str = "10 20 30 40";
    var line = try Line.read(allocator, str);
    defer line.deinit();

    try std.testing.expect(line.a.x == 10);
    try std.testing.expect(line.a.y == 20);
    try std.testing.expect(line.b.x == 30);
    try std.testing.expect(line.b.y == 40);
}

test "Environment" {
    var allocator = std.testing.allocator;
    var env = try Environment.read_from_file(allocator, "data/medium_maze.txt");
    defer env.deinit();

    try std.testing.expect(env.hero.location.x == 30);
    try std.testing.expect(env.hero.location.y == 22);
    try std.testing.expect(env.lines.len == 11);
    try std.testing.expect(env.maze_exit.x == 270);
    try std.testing.expect(env.maze_exit.y == 100);

    var lines = try allocator.alloc(*Line, 11);
    defer {
        for (lines) |l| {
            l.deinit();
        }
        allocator.free(lines);
    }
    lines[0] = try Line.init(allocator, try Point.init_coords(allocator, 5, 5), try Point.init_coords(allocator, 295, 5));
    lines[1] = try Line.init(allocator, try Point.init_coords(allocator, 295, 5), try Point.init_coords(allocator, 295, 135));
    lines[2] = try Line.init(allocator, try Point.init_coords(allocator, 295, 135), try Point.init_coords(allocator, 5, 135));
    lines[3] = try Line.init(allocator, try Point.init_coords(allocator, 5, 135), try Point.init_coords(allocator, 5, 5));
    lines[4] = try Line.init(allocator, try Point.init_coords(allocator, 241, 135), try Point.init_coords(allocator, 58, 65));
    lines[5] = try Line.init(allocator, try Point.init_coords(allocator, 114, 5), try Point.init_coords(allocator, 73, 42));
    lines[6] = try Line.init(allocator, try Point.init_coords(allocator, 130, 91), try Point.init_coords(allocator, 107, 46));
    lines[7] = try Line.init(allocator, try Point.init_coords(allocator, 196, 5), try Point.init_coords(allocator, 139, 51));
    lines[8] = try Line.init(allocator, try Point.init_coords(allocator, 219, 125), try Point.init_coords(allocator, 182, 63));
    lines[9] = try Line.init(allocator, try Point.init_coords(allocator, 267, 5), try Point.init_coords(allocator, 214, 63));
    lines[10] = try Line.init(allocator, try Point.init_coords(allocator, 271, 135), try Point.init_coords(allocator, 237, 88));

    for (lines, 0..) |line, i| {
        try std.testing.expect(line.a.x == env.lines[i].a.x);
        try std.testing.expect(line.a.y == env.lines[i].a.y);
        try std.testing.expect(line.b.x == env.lines[i].b.x);
        try std.testing.expect(line.b.y == env.lines[i].b.y);
    }
}

test {
    std.testing.refAllDecls(@This());
}
