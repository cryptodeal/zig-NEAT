pub fn LinkedList(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const Node = struct {
            prev: ?*Node = null,
            next: ?*Node = null,
            data: T,
        };
        first: ?*Node = null,
        last: ?*Node = null,
        len: usize = 0,

        pub fn push(self: *Self, node: *Node) void {
            if (self.last) |old_last| {
                old_last.next = node;
                node.prev = old_last;
                self.last = node;
            } else {
                self.first = node;
                self.last = node;
            }
            self.len += 1;
        }

        pub fn pop(self: *Self) ?*Node {
            if (self.last) |old_last| {
                self.last = old_last.prev;
                if (self.last) |new_last| {
                    new_last.next = null;
                } else {
                    self.first = null;
                }
                self.len -= 1;
                return unlink(old_last);
            } else {
                return null;
            }
        }

        pub fn shift(self: *Self) ?*Node {
            if (self.first) |old_first| {
                self.first = old_first.next;
                if (self.first) |new_first| {
                    new_first.prev = null;
                } else {
                    self.last = null;
                }
                self.len -= 1;
                return unlink(old_first);
            } else {
                return null;
            }
        }

        pub fn unshift(self: *Self, node: *Node) void {
            if (self.first) |old_first| {
                self.first = node;
                old_first.prev = node;
                node.next = old_first;
            } else {
                self.first = node;
                self.last = node;
            }
            self.len += 1;
        }

        pub fn delete(self: *Self, node: *Node) void {
            var iter: ?*Node = self.first;
            while (iter) |curr| {
                if (node.data == curr.data) {
                    if (curr == self.first) {
                        _ = self.shift();
                    } else if (curr == self.last) {
                        _ = self.pop();
                    } else {
                        curr.prev.?.next = curr.next;
                        curr.next.?.prev = curr.prev;
                        self.len -= 1;
                    }
                    return;
                } else {
                    iter = curr.next;
                }
            }
        }

        fn unlink(node: *Node) *Node {
            node.next = null;
            node.prev = null;
            return node;
        }
    };
}
