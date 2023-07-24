const std = @import("std");
const trait = @import("../trait.zig");
const math = @import("../math/math.zig");
const neat_math = @import("../math/activations.zig");
const neat_pop = @import("population.zig");
const common = @import("common.zig");
const opts = @import("../opts.zig");
const innov = @import("innovation.zig");

const logger = @constCast(opts.logger);
const Options = opts.Options;
const InnovationType = common.InnovationType;
const MutatorType = common.MutatorType;
const GenomeCompatibilityMethod = opts.GenomeCompatibilityMethod;
const Trait = trait.Trait;
const NumTraitParams = trait.NumTraitParams;
const NNode = @import("../network/nnode.zig").NNode;
const NodeNeuronType = @import("../network/common.zig").NodeNeuronType;
const Network = @import("../network/network.zig").Network;
const Gene = @import("gene.zig").Gene;
const Link = @import("../network/link.zig").Link;
const MIMOControlGene = @import("mimo_gene.zig").MIMOControlGene;
const Innovation = innov.Innovation;
const Population = neat_pop.Population;

pub const GenomeError = error{
    GenomeMissingPhenotype,
    GenomeHasNoNodes,
    GenomeHasNoGenes,
    WrongGeneCreated,
    GenomeTraitsCountMismatch,
    GenomeTraitsMismatch,
    GenomeHasNoTraits,
    GenomeHasNoTraitsOrGenes,
    GenomeNodesCountMismatch,
    GenomeNodesMismatch,
    GenomeNodesOutOfOrder,
    GenomeGenesCountMismatch,
    GenomeGenesMismatch,
    GenomeControlGenesCountMismatch,
    GenomeControlGenesMismatch,
    NetworkBuiltWithoutGenes,
    IncomingNodeNotFound,
    OutgoingNodeNotFound,
    GenomeMissingInputNode,
    GenomeMissingOutputNode,
    GenomeDuplicateGenes,
    GenomeConsecutiveGenesDisabled,
    GenomeFailedToResizeNodes,
    GenomeFailedToResizeGenes,
    GenomeHasNoTraitsOrNodes,
    GenomesHaveDifferentTraitsCount,
    NetworkMissingOutputs,
} || std.mem.Allocator.Error;

pub const Genome = struct {
    // genome id
    id: i64,
    // parameters conglomerations
    traits: []*Trait,
    // list of NNodes for network
    nodes: []*NNode,
    // list of innovation-tracking genes
    genes: []*Gene,
    // list of MIMO control genes
    control_genes: ?[]*MIMOControlGene = null,
    // allows genome to be matched w network
    phenotype: ?*Network = null,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: i64, t: []*Trait, n: []*NNode, g: []*Gene) !*Genome {
        var genome: *Genome = try allocator.create(Genome);
        genome.* = .{
            .allocator = allocator,
            .control_genes = null,
            .id = id,
            .traits = t,
            .nodes = n,
            .genes = g,
        };
        return genome;
    }

    pub fn init_modular(allocator: std.mem.Allocator, id: i64, t: []*Trait, n: []*NNode, g: []*Gene, mimo: []*MIMOControlGene) !*Genome {
        var genome: *Genome = try Genome.init(allocator, id, t, n, g);
        genome.control_genes = mimo;
        return genome;
    }

    pub fn init_rand(allocator: std.mem.Allocator, rand: std.rand.Random, new_id: i64, in: i64, out: i64, n: i64, max_hidden: i64, recurrent: bool, link_prob: f64) !*Genome {
        var total_nodes = in + out + max_hidden;
        var matrix_dim = @as(usize, @intCast(total_nodes * total_nodes));

        // init cxn matrix (will be randomized)
        var cm = try allocator.alloc(bool, matrix_dim);
        defer allocator.free(cm);

        // no nodes above this number for genome
        var max_node = in + n;
        var first_output = @as(usize, @intCast(total_nodes - out + 1));

        // init dummy trait (used in future expansion of the system)
        var new_trait = try Trait.init(allocator, NumTraitParams);
        new_trait.id = 1;

        // step through cxn matrix, randomly assigning bits
        var count: usize = 0;
        while (count < matrix_dim) : (count += 1) {
            cm[count] = rand.float(f64) < link_prob;
        }

        var nodes = std.ArrayList(*NNode).init(allocator);
        // build input nodes
        var i: usize = 1;
        while (i <= in) : (i += 1) {
            var new_node: *NNode = undefined;
            if (i < in) {
                new_node = try NNode.init(allocator, @as(i64, @intCast(i)), NodeNeuronType.InputNeuron);
            } else {
                new_node = try NNode.init(allocator, @as(i64, @intCast(i)), NodeNeuronType.BiasNeuron);
            }
            new_node.trait = new_trait;
            try nodes.append(new_node);
        }

        // build hidden nodes
        i = @as(usize, @intCast(in)) + 1;
        while (i <= in + n) : (i += 1) {
            var new_node = try NNode.init(allocator, @as(i64, @intCast(i)), NodeNeuronType.HiddenNeuron);
            new_node.trait = new_trait;
            try nodes.append(new_node);
        }

        // build the output nodes
        i = first_output;
        while (i <= total_nodes) : (i += 1) {
            var new_node = try NNode.init(allocator, @as(i64, @intCast(i)), NodeNeuronType.OutputNeuron);
            new_node.trait = new_trait;
            try nodes.append(new_node);
        }

        //
        //    i i i n n n n n n n n n n n n n n n n . . . . . . . . o o o o
        //    |                                   |                 ^     |
        //    |<----------- max_node ------------>|                 |     |
        //    |                                                     |     |
        //    |<-----------------------total_nodes -----------------|---->|
        //                                                          |
        //                                                          |
        //     first_output ----------------------------------------+
        //
        //
        var genes = std.ArrayList(*Gene).init(allocator);
        var in_node: ?*NNode = null;
        var out_node: ?*NNode = null;
        // step through cxn matrix, creating cxn genes
        count = 0;
        var flag_recurrent: bool = false;
        var col: usize = 1;
        while (col <= total_nodes) : (col += 1) {
            var row: usize = 1;
            while (row <= total_nodes) : (row += 1) {
                // only try create link if in matrix and not leading to sensor
                if (cm[count] and col > in and (col <= max_node or col >= first_output) and (row <= max_node or row >= first_output)) {
                    // if recurrent, create the cxn (gene) no matter what
                    var create_gene = true;
                    if (col > row) {
                        flag_recurrent = false;
                    } else {
                        flag_recurrent = true;
                        if (!recurrent) {
                            // skip recurrent cxns
                            create_gene = false;
                        }
                    }

                    // add new cxn (gene) to genome
                    if (create_gene) {
                        // retrieve nodes
                        i = 0;
                        while (i < nodes.items.len and (in_node == null or out_node == null)) : (i += 1) {
                            var node_id = nodes.items[i].id;
                            if (node_id == row) {
                                in_node = nodes.items[i];
                            }
                            if (node_id == col) {
                                out_node = nodes.items[i];
                            }
                        }
                        // create gene
                        var weight: f64 = @as(f64, @floatFromInt(math.rand_sign(i32, rand))) * rand.float(f64);
                        var gene = try Gene.init(allocator, weight, in_node, out_node, flag_recurrent, @as(i64, @intCast(count)), weight);

                        // add gene to genome
                        try genes.append(gene);
                    }
                }
                count += 1; // increment count
                // reset nodes
                in_node = null;
                out_node = null;
            }
        }

        var traits = try allocator.alloc(*Trait, 1);
        traits[0] = new_trait;
        return Genome.init(allocator, new_id, traits, try nodes.toOwnedSlice(), try genes.toOwnedSlice());
    }

    pub fn deinit(self: *Genome) void {
        if (self.phenotype != null) {
            self.phenotype.?.deinit();
        }
        for (self.genes) |gene| {
            gene.deinit();
        }
        self.allocator.free(self.genes);
        for (self.nodes) |node| {
            node.deinit();
        }
        self.allocator.free(self.nodes);
        for (self.traits) |t| {
            t.deinit();
        }
        if (self.control_genes != null) {
            for (self.control_genes.?) |cg| {
                cg.deinit();
            }
            self.allocator.free(self.control_genes.?);
        }
        self.allocator.free(self.traits);
        self.allocator.destroy(self);
    }

    // TODO: implement `read_genome`
    //TODO: implement `write`

    pub fn format(value: Genome, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("GENOME START\nNodes:\n", .{});
        for (value.nodes) |n| {
            var node_type: []const u8 = undefined;
            switch (n.neuron_type) {
                NodeNeuronType.InputNeuron => node_type = "I",
                NodeNeuronType.OutputNeuron => node_type = "O",
                NodeNeuronType.BiasNeuron => node_type = "B",
                NodeNeuronType.HiddenNeuron => node_type = "H",
            }
            try writer.print("\t{s}{any} \n", .{ node_type, n });
        }
        try writer.print("Genes:\n", .{});
        for (value.genes) |gn| {
            try writer.print("\t{any}\n", .{gn});
        }
        try writer.print("Traits:\n", .{});
        for (value.traits) |t| {
            try writer.print("\t{any}\n", .{t});
        }
        return writer.print("GENOME END", .{});
    }

    /// returns count of non-disabled genes
    pub fn extrons(self: *Genome) i64 {
        var total: i64 = 0;
        for (self.genes) |gene| {
            if (gene.is_enabled) {
                total += 1;
            }
        }
        return total;
    }

    pub fn is_equal(self: *Genome, og: *Genome) !bool {
        if (self.traits.len != og.traits.len) {
            logger.err("traits count mismatch: {d} != {d}", .{ self.traits.len, og.traits.len }, @src());
            return GenomeError.GenomeTraitsCountMismatch;
        }
        for (og.traits, 0..) |tr, i| {
            if (!tr.is_equal(self.traits[i])) {
                logger.err("traits mismatch, expected: {any}, but found: {any}", .{ tr, self.traits[i] }, @src());
                return GenomeError.GenomeTraitsMismatch;
            }
        }
        if (self.nodes.len != og.nodes.len) {
            logger.err("nodes count mismatch: {d} != {d}", .{ self.nodes.len, og.nodes.len }, @src());
            return GenomeError.GenomeNodesCountMismatch;
        }
        for (og.nodes, 0..) |nd, i| {
            if (!nd.is_equal(self.nodes[i])) {
                logger.err("nodes mismatch, expected: {any}\n, but found: {any}", .{ nd, self.nodes[i] }, @src());
                return GenomeError.GenomeNodesMismatch;
            }
        }
        if (self.genes.len != og.genes.len) {
            logger.err("genes count mismatch: {d} != {d}", .{ self.genes.len, og.genes.len }, @src());
            return GenomeError.GenomeGenesCountMismatch;
        }
        for (og.genes, 0..) |gen, i| {
            if (!gen.is_equal(self.genes[i])) {
                logger.err("genes mismatch, expected: {any}\n, but found: {any}", .{ gen, self.nodes[i] }, @src());
                return GenomeError.GenomeGenesMismatch;
            }
        }
        if ((self.control_genes == null and og.control_genes != null) or (self.control_genes != null and og.control_genes == null)) {
            logger.err("control genes mismatch: {any} != {any}", .{ self.control_genes, og.control_genes }, @src());
            return GenomeError.GenomeControlGenesMismatch;
        }
        if (self.control_genes != null and og.control_genes != null) {
            if (self.control_genes.?.len != og.control_genes.?.len) {
                logger.err("control genes count mismatch: {d} != {d}", .{ self.control_genes.?.len, og.control_genes.?.len }, @src());
                return GenomeError.GenomeControlGenesCountMismatch;
            }
            for (og.control_genes.?, 0..) |cg, i| {
                if (!cg.is_equal(self.control_genes.?[i])) {
                    logger.err("control genes mismatch, expected: {any}\n, but found: {any}", .{ cg, self.control_genes.?[i] }, @src());
                    return GenomeError.GenomeControlGenesMismatch;
                }
            }
        }

        return true;
    }

    pub fn get_last_node_id(self: *Genome) !i64 {
        if (self.nodes.len == 0) {
            logger.err("Genome has no nodes", .{}, @src());
            return GenomeError.GenomeHasNoNodes;
        }
        var id: i64 = self.nodes[self.nodes.len - 1].id;
        if (self.control_genes != null) {
            for (self.control_genes.?) |cg| {
                if (cg.control_node.id > id) {
                    id = cg.control_node.id;
                }
            }
        }
        return id;
    }

    pub fn get_next_gene_innov_num(self: *Genome) !i64 {
        var inn_num: i64 = 0;
        if (self.genes.len > 0) {
            inn_num = self.genes[self.genes.len - 1].innovation_num;
        } else {
            logger.err("Genome has no genes", .{}, @src());
            return GenomeError.GenomeHasNoGenes;
        }
        // check control genes (if any)
        if (self.control_genes != null and self.control_genes.?.len > 0) {
            var c_inn_num: i64 = self.control_genes.?[self.control_genes.?.len - 1].innovation_num;
            if (c_inn_num > inn_num) {
                inn_num = c_inn_num;
            }
        }
        return inn_num + 1;
    }

    pub fn has_node(self: *Genome, node: ?*NNode) !bool {
        if (node == null) {
            return false;
        }
        var id = try self.get_last_node_id();
        if (node.?.id > id) {
            return false;
        }
        for (self.nodes) |n| {
            if (n.id == node.?.id) {
                return true;
            }
        }
        return false;
    }

    pub fn has_gene(self: *Genome, gene: *Gene) !bool {
        var inn = try self.get_next_gene_innov_num();
        if (gene.innovation_num > inn) {
            // gene has innovation num greater than found in this genome;
            // this means that this gene is not from this genome lineage
            return false;
        }
        // find genetically equal link in this genome to the provided gene
        for (self.genes) |g| {
            if (g.link.is_genetically_eql(gene.link)) {
                return true;
            }
        }
        return false;
    }

    pub fn genesis(self: *Genome, allocator: std.mem.Allocator, net_id: i64) !*Network {
        var in_list = std.ArrayList(*NNode).init(allocator);
        var out_list = std.ArrayList(*NNode).init(allocator);
        var all_list = std.ArrayList(*NNode).init(allocator);

        var new_node: *NNode = undefined;
        // create the network nodes
        for (self.nodes) |n| {
            new_node = try NNode.init_copy(allocator, n, n.trait);
            if (n.neuron_type == NodeNeuronType.InputNeuron or n.neuron_type == NodeNeuronType.BiasNeuron) {
                try in_list.append(new_node);
            } else if (n.neuron_type == NodeNeuronType.OutputNeuron) {
                try out_list.append(new_node);
            }
            try all_list.append(new_node);
            n.phenotype_analogue = new_node;
        }

        if (self.genes.len == 0) {
            logger.err("network built without GENES; the result can be unpredictable", .{}, @src());
            return GenomeError.NetworkBuiltWithoutGenes;
        }

        if (out_list.items.len == 0) {
            logger.err("network without OUTPUTS; the result can be unpredictable. Genome: {any}", .{self}, @src());
            return GenomeError.NetworkMissingOutputs;
        }

        var in_node: *NNode = undefined;
        var out_node: *NNode = undefined;
        var cur_link: *Link = undefined;
        var new_link: *Link = undefined;
        // walk through genes, creating links
        for (self.genes) |gn| {
            if (gn.is_enabled) {
                cur_link = gn.link;
                in_node = cur_link.in_node.?.phenotype_analogue;
                out_node = cur_link.out_node.?.phenotype_analogue;
                // NOTE: This line could be run through a recurrence check if desired
                // (no need to in the current implementation of NEAT)
                new_link = try Link.init_with_trait(allocator, cur_link.trait, cur_link.cxn_weight, in_node, out_node, cur_link.is_recurrent);

                // add link to the connected nodes
                try out_node.incoming.append(new_link);
                try in_node.outgoing.append(new_link);
            }
        }

        var new_net: *Network = undefined;
        if (self.control_genes == null or self.control_genes.?.len == 0) {
            new_net = try Network.init(allocator, try in_list.toOwnedSlice(), try out_list.toOwnedSlice(), try all_list.toOwnedSlice(), net_id);
        } else {
            // create MIMO control genes
            var c_nodes = std.ArrayList(*NNode).init(allocator);
            for (self.control_genes.?) |cg| {
                if (cg.is_enabled) {
                    var new_copy_node = try NNode.init_copy(allocator, cg.control_node, cg.control_node.trait);
                    // connect inputs
                    for (cg.control_node.incoming.items) |l| {
                        in_node = l.in_node.?.phenotype_analogue;
                        out_node = new_copy_node;
                        new_link = try Link.init(allocator, l.cxn_weight, in_node, out_node, false);
                        // only incoming to control node
                        try out_node.incoming.append(new_link);
                    }
                    // connect outputs
                    for (cg.control_node.outgoing.items) |l| {
                        in_node = new_copy_node;
                        out_node = l.out_node.?.phenotype_analogue;
                        new_link = try Link.init(allocator, l.cxn_weight, in_node, out_node, false);
                        // only outgoing from control node
                        try in_node.outgoing.append(new_link);
                    }

                    // store control node
                    try c_nodes.append(new_copy_node);
                }
            }

            new_net = try Network.init_modular(allocator, try in_list.toOwnedSlice(), try out_list.toOwnedSlice(), try all_list.toOwnedSlice(), try c_nodes.toOwnedSlice(), net_id);
        }
        // free memory before initializing updated Graph
        if (self.phenotype != null) {
            self.phenotype.?.deinit();
        }
        self.phenotype = new_net;
        return new_net;
    }

    pub fn duplicate(self: *Genome, allocator: std.mem.Allocator, new_id: i64) !*Genome {
        // duplicate the traits
        var traits_dup = try allocator.alloc(*Trait, self.traits.len);
        for (self.traits, 0..) |tr, i| {
            traits_dup[i] = try Trait.new_trait_copy(allocator, tr);
        }

        // duplicate NNodes
        var nodes_dup = try allocator.alloc(*NNode, self.nodes.len);
        for (self.nodes, 0..) |nd, i| {
            // find duplicate of the trait node points to
            var assoc_trait = nd.trait;
            if (assoc_trait != null) {
                assoc_trait = common.trait_with_id(assoc_trait.?.id.?, traits_dup);
            }
            nodes_dup[i] = try NNode.init_copy(allocator, nd, assoc_trait);
        }

        // duplicate Genes
        var genes_dup = try allocator.alloc(*Gene, self.genes.len);
        for (self.genes, 0..) |gn, i| {
            // find nodes connected by gene's link
            var in_node = common.node_with_id(gn.link.in_node.?.id, nodes_dup);
            if (in_node == null) {
                logger.err("incoming node: {d} not found for gene {any}", .{ gn.link.in_node.?.id, gn }, @src());
                return GenomeError.IncomingNodeNotFound;
            }
            var out_node = common.node_with_id(gn.link.out_node.?.id, nodes_dup);
            if (out_node == null) {
                logger.err("outgoing node: {d} not found for gene {any}", .{ gn.link.out_node.?.id, gn }, @src());
                return GenomeError.OutgoingNodeNotFound;
            }
            // Find the duplicate of trait associated with this gene
            var assoc_trait = gn.link.trait;
            if (assoc_trait != null) {
                assoc_trait = common.trait_with_id(assoc_trait.?.id.?, traits_dup);
            }
            genes_dup[i] = try Gene.init_copy(allocator, gn, assoc_trait, in_node, out_node);
        }

        if (self.control_genes == null or self.control_genes.?.len == 0) {
            // If no MIMO control genes return plain genome
            return Genome.init(allocator, new_id, traits_dup, nodes_dup, genes_dup);
        } else {
            // Duplicate MIMO Control Genes and build modular genome
            var control_genes_dup = try allocator.alloc(*MIMOControlGene, self.control_genes.?.len);
            for (self.control_genes.?, 0..) |cg, i| {
                // duplicate control node
                var control_node: *NNode = cg.control_node;
                // find duplicate of trait associated w control node
                var assoc_trait = control_node.trait;
                if (assoc_trait != null) {
                    assoc_trait = common.trait_with_id(assoc_trait.?.id.?, traits_dup);
                }
                var node_copy = try NNode.init_copy(allocator, control_node, assoc_trait);
                // add incoming links
                for (control_node.incoming.items) |l| {
                    var in_node = common.node_with_id(l.in_node.?.id, nodes_dup);
                    if (in_node == null) {
                        std.debug.print("incoming node: {d} not found for control node: {d}", .{ l.in_node.?.id, control_node.id });
                        return GenomeError.IncomingNodeNotFound;
                    }
                    var new_in_link = try Link.init_copy(allocator, l, in_node.?, node_copy);
                    try node_copy.incoming.append(new_in_link);
                }
                // add outgoing links
                for (control_node.outgoing.items) |l| {
                    var out_node = common.node_with_id(l.out_node.?.id, nodes_dup);
                    if (out_node == null) {
                        std.debug.print("outgoing node: {d} not found for control node: {d}", .{ l.out_node.?.id, control_node.id });
                        return GenomeError.OutgoingNodeNotFound;
                    }
                    var new_out_link = try Link.init_copy(allocator, l, node_copy, out_node.?);
                    try node_copy.outgoing.append(new_out_link);
                }

                // add MIMO control gene
                control_genes_dup[i] = try MIMOControlGene.init_from_copy(allocator, cg, node_copy);
            }
            return Genome.init_modular(allocator, new_id, traits_dup, nodes_dup, genes_dup, control_genes_dup);
        }
    }

    pub fn verify(self: *Genome) !bool {
        if (self.genes.len == 0) {
            std.debug.print("Genome has no Genes", .{});
            return GenomeError.GenomeHasNoGenes;
        }
        if (self.nodes.len == 0) {
            std.debug.print("Genome has no Nodes", .{});
            return GenomeError.GenomeHasNoNodes;
        }
        if (self.traits.len == 0) {
            std.debug.print("Genome has no Traits", .{});
            return GenomeError.GenomeHasNoTraits;
        }

        // check each Gene's nodes
        for (self.genes) |gn| {
            var in_node = gn.link.in_node.?;
            var out_node = gn.link.out_node.?;
            var input_found = false;
            var out_found = false;
            var i: usize = 0;
            while (i < self.nodes.len and (!input_found or !out_found)) : (i += 1) {
                if (in_node.id == self.nodes[i].id) {
                    input_found = true;
                }
                if (out_node.id == self.nodes[i].id) {
                    out_found = true;
                }
            }

            // check results
            if (!input_found) {
                std.debug.print("missing input node of gene in the genome nodes list", .{});
                return GenomeError.GenomeMissingInputNode;
            }
            if (!out_found) {
                std.debug.print("missing output node of gene in the genome nodes list", .{});
                return GenomeError.GenomeMissingOutputNode;
            }
        }
        // validate node's in order
        var last_id: i64 = 0;
        for (self.nodes) |n| {
            if (n.id < last_id) {
                std.debug.print("nodes out of order in genome", .{});
                return GenomeError.GenomeNodesOutOfOrder;
            }
            last_id = n.id;
        }

        // validate no duplicate Genes
        for (self.genes) |gn| {
            for (self.genes) |gn2| {
                if (!std.meta.eql(gn, gn2) and gn.link.is_genetically_eql(gn2.link)) {
                    std.debug.print("duplicate genes found: {any} == {any}", .{ gn, gn2 });
                    return GenomeError.GenomeDuplicateGenes;
                }
            }
        }

        // check for consecutive disabled
        // N.B. this isn't necessarily bad... (but it's probably not ideal)
        if (self.nodes.len > 500) {
            var disabled = false;
            for (self.genes) |gn| {
                if (!gn.is_enabled and disabled) {
                    std.debug.print("two gene disables in a row", .{});
                    return GenomeError.GenomeConsecutiveGenesDisabled;
                }
                disabled = !gn.is_enabled;
            }
        }

        return true;
    }

    pub fn node_insert(_: *Genome, allocator: std.mem.Allocator, nodes: []*NNode, n: ?*NNode) ![]*NNode {
        var used_nodes = nodes;
        if (n == null) {
            logger.warn("GENOME: attempting to insert NIL node into genome nodes, recovered\n", .{}, @src());
            return used_nodes;
        }
        var idx: usize = used_nodes.len;
        // quick insert @ beginning or end
        if (idx == 0 or n.?.id >= used_nodes[idx - 1].id) {
            if (allocator.resize(used_nodes, idx + 1)) {
                used_nodes.len = idx + 1;
                // append last
                used_nodes[used_nodes.len - 1] = n.?;
                return used_nodes;
            }
            var tmp_nodes = std.ArrayList(*NNode).fromOwnedSlice(allocator, used_nodes);
            try tmp_nodes.append(n.?);
            return tmp_nodes.toOwnedSlice();
        } else if (n.?.id <= used_nodes[0].id) {
            // insert first
            idx = 0;
        }
        // find split idx
        var i: usize = idx - 1;
        while (i >= 0) : (i -= 1) {
            if (n.?.id == used_nodes[i].id) {
                idx = i;
                break;
            } else if (n.?.id > used_nodes[i].id) {
                idx = i + 1;
                break;
            }
        }
        var new_nodes = std.ArrayList(*NNode).init(allocator);
        try new_nodes.appendSlice(used_nodes[0..idx]);
        try new_nodes.append(n.?);
        try new_nodes.appendSlice(used_nodes[idx..]);
        allocator.free(used_nodes);
        return new_nodes.toOwnedSlice();
    }

    pub fn gene_insert(_: *Genome, allocator: std.mem.Allocator, genes: []*Gene, g: ?*Gene) ![]*Gene {
        var used_genes = genes;
        if (g == null) {
            logger.warn("GENOME: attempting to insert null Gene into genome Genes, recovered\n", .{}, @src());
            return used_genes;
        }
        var idx: usize = used_genes.len;
        // quick insert @ beginning or end
        if (idx == 0 or g.?.innovation_num >= used_genes[idx - 1].innovation_num) {
            if (allocator.resize(used_genes, idx + 1)) {
                used_genes.len = idx + 1;
                // append last
                used_genes[used_genes.len - 1] = g.?;
                return used_genes;
            }
            var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, used_genes);
            try tmp_genes.append(g.?);
            return tmp_genes.toOwnedSlice();
        } else if (g.?.innovation_num <= used_genes[0].innovation_num) {
            // insert first
            idx = 0;
        }
        // find split idx
        var i: usize = idx - 1;
        while (i >= 0) : (i -= 1) {
            if (g.?.innovation_num == used_genes[i].innovation_num) {
                idx = i;
                break;
            } else if (g.?.innovation_num > used_genes[i].innovation_num) {
                idx = i + 1;
                break;
            }
        }

        var new_genes = std.ArrayList(*Gene).init(allocator);
        try new_genes.appendSlice(used_genes[0..idx]);
        try new_genes.append(g.?);
        try new_genes.appendSlice(used_genes[idx..]);
        allocator.free(used_genes);
        return new_genes.toOwnedSlice();
    }

    pub fn compatability(self: *Genome, og: *Genome, opt: *Options) f64 {
        if (opt.gen_compat_method == GenomeCompatibilityMethod.GenomeCompatibilityMethodLinear) {
            return self.compat_linear(og, opt);
        } else {
            return self.compat_fast(og, opt);
        }
    }

    pub fn compat_linear(self: *Genome, og: *Genome, opt: *Options) f64 {
        var num_disjoint: f64 = 0.0;
        var num_excess: f64 = 0.0;
        var mut_diff_total: f64 = 0.0;
        var num_matching: f64 = 0.0;
        var size_1 = self.genes.len;
        var size_2 = og.genes.len;
        var max_genome_size = size_2;
        if (size_1 > size_2) {
            max_genome_size = size_1;
        }
        var gene_1: *Gene = undefined;
        var gene_2: *Gene = undefined;
        var i: usize = 0;
        var i_1: usize = 0;
        var i_2: usize = 0;
        while (i < max_genome_size) : (i += 1) {
            if (i_1 >= size_1) {
                num_excess += 1.0;
                i_2 += 1;
            } else if (i_2 >= size_2) {
                num_excess += 1.0;
                i_1 += 1;
            } else {
                gene_1 = self.genes[i_1];
                gene_2 = og.genes[i_2];
                var p1_innov = gene_1.innovation_num;
                var p2_innov = gene_2.innovation_num;

                if (p1_innov == p2_innov) {
                    num_matching += 1.0;
                    var mut_diff = @fabs(gene_1.mutation_num - gene_2.mutation_num);
                    mut_diff_total += mut_diff;
                    i_1 += 1;
                    i_2 += 1;
                } else if (p1_innov < p2_innov) {
                    i_1 += 1;
                    num_disjoint += 1.0;
                } else if (p2_innov < p1_innov) {
                    i_2 += 1;
                    num_disjoint += 1.0;
                }
            }
        }
        var comp = opt.disjoint_coeff * num_disjoint + opt.excess_coeff * num_excess + opt.mut_diff_coeff * (mut_diff_total / num_matching);
        return comp;
    }

    pub fn compat_fast(self: *Genome, og: *Genome, opt: *Options) f64 {
        var list1_count = self.genes.len;
        var list2_count = og.genes.len;
        // test edge cases
        if (list1_count == 0 and list2_count == 0) {
            // both lists are empty; no disparities so genomes are compatible
            return 0.0;
        }
        if (list1_count == 0) {
            // all list2 genes are excess
            return @as(f64, @floatFromInt(list2_count)) * opt.excess_coeff;
        }
        if (list2_count == 0) {
            // all list1 genes are excess
            return @as(f64, @floatFromInt(list1_count)) * opt.excess_coeff;
        }

        var excess_genes_switch: usize = 0;
        var num_matching: usize = 0;
        var compat: f64 = 0.0;
        var mut_diff: f64 = 0.0;
        var list1_idx: i64 = @as(i64, @intCast(list1_count)) - 1;
        var list2_idx: i64 = @as(i64, @intCast(list2_count)) - 1;
        var gene_1: *Gene = self.genes[@as(usize, @intCast(list1_idx))];
        var gene_2: *Gene = og.genes[@as(usize, @intCast(list2_idx))];
        while (true) {
            if (gene_2.innovation_num > gene_1.innovation_num) {
                // most common test cases @ top for perf
                if (excess_genes_switch == 3) {
                    // no more excess genes; mismatch must be disjoint
                    compat += opt.disjoint_coeff;
                } else if (excess_genes_switch == 2) {
                    // another excess gene on genome 2
                    compat += opt.excess_coeff;
                } else if (excess_genes_switch == 1) {
                    // found first non-excess gene
                    excess_genes_switch = 3;
                    compat += opt.disjoint_coeff;
                } else {
                    // first gene is excess and on genome 2
                    excess_genes_switch = 2;
                    compat += opt.excess_coeff;
                }
                // move to next gene in list2
                list2_idx -= 1;
            } else if (gene_1.innovation_num == gene_2.innovation_num) {
                // no more excess genes
                // for perf, faster to set this every time than test if `excess_genes_switch == 3`
                excess_genes_switch = 3;

                // matching genes; increase compat by `mutation_num difference * coeff`
                mut_diff += @fabs(gene_1.mutation_num - gene_2.mutation_num);
                num_matching += 1;

                // move to next gene in both lists
                list1_idx -= 1;
                list2_idx -= 1;
            } else {
                // most common test cases @ top for perf
                if (excess_genes_switch == 3) {
                    // no more excess genes; mismatch must be disjoint
                    compat += opt.disjoint_coeff;
                } else if (excess_genes_switch == 1) {
                    // another excess gene on genome 1
                    compat += opt.excess_coeff;
                } else if (excess_genes_switch == 2) {
                    // found the first non-excess gene
                    excess_genes_switch = 3;
                    compat += opt.disjoint_coeff;
                } else {
                    // first gene is excess; gene belongs to genome 1
                    excess_genes_switch = 1;
                    compat += opt.excess_coeff;
                }
                // move to next gene in list1
                list1_idx -= 1;
            }

            // check if reached end of either list
            if (list1_idx < 0) {
                // all remaining list2 genes are disjoint
                compat += @as(f64, @floatFromInt(list2_idx + 1)) * opt.disjoint_coeff;
                break;
            }
            if (list2_idx < 0) {
                // all remaining list2 genes are disjoint
                compat += @as(f64, @floatFromInt(list1_idx + 1)) * opt.disjoint_coeff;
                break;
            }
            gene_1 = self.genes[@as(usize, @intCast(list1_idx))];
            gene_2 = og.genes[@as(usize, @intCast(list2_idx))];
        }
        if (num_matching > 0) {
            compat += mut_diff * opt.mut_diff_coeff / @as(f64, @floatFromInt(num_matching));
        }
        return compat;
    }

    pub fn mutate_connect_sensors(self: *Genome, allocator: std.mem.Allocator, rand: std.rand.Random, innovations: *Population, _: *Options) !bool {
        if (self.genes.len == 0) {
            logger.debug("GENOME ID: {d} ---- mutate connect sensors FAILED; Genome has no Genes!", .{self.id}, @src());
            return false;
        }

        // find all sensors/outputs
        var sensors = std.ArrayList(*NNode).init(allocator);
        defer sensors.deinit();
        var outputs = std.ArrayList(*NNode).init(allocator);
        defer outputs.deinit();
        for (self.nodes) |n| {
            if (n.is_sensor()) {
                try sensors.append(n);
            } else {
                try outputs.append(n);
            }
        }

        // find sensors not connected (if any)
        var disconnected_sensors = std.ArrayList(*NNode).init(allocator);
        defer disconnected_sensors.deinit();
        for (sensors.items) |sensor| {
            var connected = false;
            // iterate over genes and count number of output cxns from given sensor
            for (self.genes) |gene| {
                if (gene.link.in_node.?.id == sensor.id) {
                    connected = true;
                    break;
                }
            }
            if (!connected) {
                // store disconnected sensor
                try disconnected_sensors.append(sensor);
            }
        }

        // if all sensors are connected - stop
        if (disconnected_sensors.items.len == 0) {
            logger.debug("GENOME ID: {d} ---- mutate connect sensors FAILED; all sensors are connected!", .{self.id}, @src());
            return false;
        }

        // pick randomly from disconnected sensors
        var sensor = disconnected_sensors.items[rand.uintLessThan(usize, disconnected_sensors.items.len)];
        // add new links to chosen sensor, avoid duplicates
        var link_added = false;
        for (outputs.items) |output| {
            var found = false;
            for (self.genes) |gene| {
                if (gene.link.in_node.? == sensor and gene.link.out_node == output) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                var gene: ?*Gene = null;
                // check whether this innovation already occurred in population
                var innovation_found = false;
                for (innovations.innovations.items) |inn| {
                    if (inn.innovation_type == InnovationType.NewLinkInnType and inn.in_node_id == sensor.id and inn.out_node_id == output.id and !inn.is_recurrent) {
                        gene = try Gene.init_with_trait(allocator, self.traits[inn.new_trait_num], inn.new_weight, sensor, output, false, inn.innovation_num, 0);
                        innovation_found = true;
                        break;
                    }
                }

                // the innovation is totally novel
                if (!innovation_found) {
                    // choose a random trait
                    var trait_num = rand.uintLessThan(usize, self.traits.len);
                    // choose the new weight
                    var new_weight = @as(f64, @floatFromInt(math.rand_sign(i32, rand))) * rand.float(f64) * 10.0;
                    // read next innovation id
                    var next_innov_id = innovations.get_next_innovation_number();

                    // create new gene
                    gene = try Gene.init_with_trait(allocator, self.traits[trait_num], new_weight, sensor, output, false, next_innov_id, new_weight);

                    // add the innovation for created link
                    var new_innovation = try Innovation.init_for_link(allocator, sensor.id, output.id, next_innov_id, new_weight, trait_num);
                    try innovations.store_innovation(new_innovation);
                } else if (gene != null and try self.has_gene(gene.?)) {
                    logger.info("GENOME: Connect sensors innovation found [{any}] in the same genome [{d}] for gene: {any}\n{any}", .{ innovation_found, self.id, gene, self }, @src());
                    return false;
                }

                // add the new Gene to the Genome
                if (gene != null) {
                    self.genes = try self.gene_insert(allocator, self.genes, gene.?);
                    link_added = true;
                }
            }
        }
        return link_added;
    }

    pub fn mutate_add_link(self: *Genome, allocator: std.mem.Allocator, rand: std.rand.Random, innovations: *Population, opt: *Options) !bool {
        var nodes_len = self.nodes.len;
        if (self.phenotype == null) {
            logger.debug("GENOME ID: {d} ---- mutate add link FAILED; cannot add link to Genome missing phenotype (Network)!", .{self.id}, @src());
            return GenomeError.GenomeMissingPhenotype;
        } else if (nodes_len == 0) {
            logger.debug("GENOME ID: {d} ---- mutate add link FAILED; cannot add link to Genome missing nodes (NNode)!", .{self.id}, @src());
            return GenomeError.GenomeHasNoNodes;
        }

        // decide whether to make link recurrent
        var do_recur = false;
        if (rand.float(f64) < opt.recur_only_prob) {
            do_recur = true;
        }

        // find first non-sensor so the `to-node` won't look at sensors as destination
        var first_non_sensor: usize = 0;
        for (self.nodes) |n| {
            if (n.is_sensor()) first_non_sensor += 1 else break;
        }

        // tracks # of attempts to find unconnected pair
        var try_count: usize = 0;

        // iterate over nodes and try to add new link
        var node1: ?*NNode = null;
        var node2: ?*NNode = null;
        var found = false;
        while (try_count < opt.new_link_tries) {
            var node_num1: usize = 0;
            var node_num2: usize = 0;
            if (do_recur) {
                // 50% of prob to decide create a recurrent link (node X to node X)
                // 50% of a normal link (node X to node Y)
                var loop_recur = false;
                if (rand.float(f64) > 0.5) {
                    loop_recur = true;
                }
                if (loop_recur) {
                    node_num1 = first_non_sensor + rand.uintLessThan(usize, nodes_len - first_non_sensor); // only non-sensors
                    node_num2 = node_num1;
                } else {
                    while (node_num1 == node_num2) {
                        node_num1 = rand.uintLessThan(usize, nodes_len);
                        node_num2 = first_non_sensor + rand.uintLessThan(usize, nodes_len - first_non_sensor); // only NON SENSOR
                    }
                }
            } else {
                while (node_num1 == node_num2) {
                    node_num1 = rand.uintLessThan(usize, nodes_len);
                    node_num2 = first_non_sensor + rand.uintLessThan(usize, nodes_len - first_non_sensor); // only NON SENSOR
                }
            }
            // get corresponding nodes
            node1 = self.nodes[node_num1];
            node2 = self.nodes[node_num2];

            // check if link already exists (ALSO STOP @ END OF GENES!!!!)
            var link_exists = false;
            if (node2.?.is_sensor()) {
                // don't allow SENSORS to get input
                link_exists = true;
            } else {
                for (self.genes) |gene| {
                    if (gene.link.in_node.?.id == node1.?.id and gene.link.out_node.?.id == node2.?.id and gene.link.is_recurrent == do_recur) {
                        // link already exists
                        link_exists = true;
                        break;
                    }
                }
            }

            if (!link_exists) {
                var thresh: i64 = @intCast(nodes_len * nodes_len);
                var count: i64 = 0;
                var recur_flag = self.phenotype.?.is_recurrent(node1.?.phenotype_analogue, node2.?.phenotype_analogue, &count, thresh);
                if (count > thresh) {
                    logger.debug("GENOME: LOOP DETECTED DURING A RECURRENCY CHECK -> node in: {any} <-> node out: {any}", .{ node1.?.phenotype_analogue, node2.?.phenotype_analogue }, @src());
                }

                // Make sure it finds the right kind of link (recurrent or not)
                if ((!recur_flag and do_recur) or (recur_flag and !do_recur)) {
                    try_count += 1;
                } else {
                    // the open link found
                    try_count = opt.new_link_tries;
                    found = true;
                }
            } else {
                try_count += 1;
            }
        }

        // continue only if open link was found & corresponding nodes were set
        if (node1 != null and node2 != null and found) {
            var gene: ?*Gene = null;
            // check whether innovaton already occurred in population
            var innovation_found = false;
            for (innovations.innovations.items) |inn| {
                // match the innovation in innovation list
                if (inn.innovation_type == InnovationType.NewLinkInnType and inn.in_node_id == node1.?.id and inn.out_node_id == node2.?.id and inn.is_recurrent == do_recur) {
                    // Create new gene
                    gene = try Gene.init_with_trait(allocator, self.traits[inn.new_trait_num], inn.new_weight, node1.?, node2.?, do_recur, inn.innovation_num, 0);
                    innovation_found = true;
                    break;
                }
            }

            // innovation is totally novel
            if (!innovation_found) {
                // choose random trait
                var trait_num: usize = rand.uintLessThan(usize, self.traits.len);
                // choose new weight
                var new_weight = @as(f64, @floatFromInt(math.rand_sign(i32, rand))) * rand.float(f64) * 10.0;
                // read next innovation id
                var next_innov_id = innovations.get_next_innovation_number();

                // create new gene
                gene = try Gene.init_with_trait(allocator, self.traits[trait_num], new_weight, node1.?, node2.?, do_recur, next_innov_id, new_weight);

                // add the innovation
                var innovation = try Innovation.init_for_recurrent_link(allocator, node1.?.id, node2.?.id, next_innov_id, new_weight, trait_num, do_recur);
                try innovations.store_innovation(innovation);
            } else if (gene != null and try self.has_gene(gene.?)) {
                logger.info("GENOME: Mutate add link innovation found [{any}] in the same genome [{d}] for gene: {any}\n{any}\n", .{ innovation_found, self.id, gene.?, self }, @src());
                return false;
            }

            // sanity check
            if (gene != null and gene.?.link.in_node.?.id == gene.?.link.out_node.?.id and !do_recur) {
                logger.warn("Recurent link created when recurency is not enabled: {any}", .{gene}, @src());
                logger.err("wrong gene created\n{any}", .{self}, @src());
                return GenomeError.WrongGeneCreated;
            }

            // add the new Gene to Genome
            if (gene != null) {
                // modifying Genome's gene list; use Genome's allocator
                self.genes = try self.gene_insert(self.allocator, self.genes, gene);
            }
        }
        return found;
    }

    pub fn mutate_add_node(self: *Genome, allocator: std.mem.Allocator, rand: std.rand.Random, innovations: *Population, opt: *Options) !bool {
        // it's possible to have network without any link
        if (self.genes.len == 0) {
            return false;
        }

        // first, find a random Gene already in Genome
        var found = false;
        var gene: ?*Gene = null;

        // for very small genomes, we need to bias splitting towards older links to avoid a "chaining"
        // effect, which is likely to occur when we keep splitting between the same two nodes repeatedly
        if (self.genes.len < 15) {
            for (self.genes) |gn| {
                // randomize which gene is selected
                if (gn.is_enabled and gn.link.in_node.?.neuron_type != NodeNeuronType.BiasNeuron and rand.float(f32) >= 0.3) {
                    gene = gn;
                    found = true;
                    break;
                }
            }
        } else {
            var try_count: usize = 0;
            // alternative uniform random choice of genes. When the genome is not tiny, it is safe to choose randomly.
            while (try_count < 20 and !found) {
                var gene_num = rand.uintLessThan(usize, self.genes.len);
                gene = self.genes[gene_num];
                if (gene.?.is_enabled and gene.?.link.in_node.?.neuron_type != NodeNeuronType.BiasNeuron) {
                    found = true;
                }
                try_count += 1;
            }
        }
        if (!found or gene == null) {
            // failed to find appropriate gene
            logger.debug("GENOME ID: {d} ---- mutate add node FAILED; unable to find appropriate gene!", .{self.id}, @src());
            return false;
        }

        gene.?.is_enabled = false;

        // extract link
        var link = gene.?.link;
        // extract weight
        var old_weight = link.cxn_weight;
        // get old link's trait
        var _trait = link.trait;

        if (link.in_node == null or link.out_node == null) {
            // anomalous Link found missing In or Out Node
            logger.debug("GENOME ID: {d} ---- mutate add node FAILED; Anomalous link found with either IN or OUT node not set!", .{self.id}, @src());
            return false;
        }

        // extract the nodes
        var in_node = link.in_node.?;
        var out_node = link.out_node.?;

        var gene1: ?*Gene = null;
        var gene2: ?*Gene = null;
        var node: ?*NNode = null;

        // check whether this innovation already occurred in the population
        var innovation_found = false;
        for (innovations.innovations.items) |inn| {
            // We check to see if an innovation already occurred that was:
            //   - A new node
            //   - Stuck between the same nodes as were chosen for this mutation
            //   - Splitting the same gene as chosen for this mutation
            // If so, we know this mutation is not a novel innovation in this generation
            // so we make it match the original, identical mutation which occurred
            // elsewhere in the population by coincidence
            if (inn.innovation_type == InnovationType.NewNodeInnType and inn.in_node_id == in_node.id and inn.out_node_id == out_node.id and inn.old_innov_num == gene.?.innovation_num) {
                // create the new Node
                node = try NNode.init(allocator, inn.new_node_id, NodeNeuronType.HiddenNeuron);
                // by convention, it will point to the first trait
                // N.B. in future, may want to change this
                node.?.trait = self.traits[0];

                // create the new Genes
                gene1 = try Gene.init_with_trait(allocator, _trait, 1.0, in_node, node.?, link.is_recurrent, inn.innovation_num, 0);
                gene2 = try Gene.init_with_trait(allocator, _trait, old_weight, node.?, out_node, false, inn.innnovation_num2, 0);

                innovation_found = true;
                break;
            }
        }

        // innovation is totally novel
        if (!innovation_found) {
            // get current node id post increment
            var new_node_id = innovations.get_next_node_id();

            // create the new NNode
            node = try NNode.init(allocator, new_node_id, NodeNeuronType.HiddenNeuron);
            // by convention, it will point to the first trait
            node.?.trait = self.traits[0];
            // set node activation function as random from a list of types registered with opts
            var activation_type = try opt.random_node_activation_type(rand);
            node.?.activation_type = activation_type;

            // get next innovation id for gene 1
            var gene1_innovation = innovations.get_next_innovation_number();
            // create gene with the current gene innovation
            gene1 = try Gene.init_with_trait(allocator, _trait, 1.0, in_node, node.?, link.is_recurrent, gene1_innovation, 0);

            // get the next innovation id for gene 2
            var gene2_innovation = innovations.get_next_innovation_number();
            // create the second gene with this innovation incremented
            gene2 = try Gene.init_with_trait(allocator, _trait, old_weight, node.?, out_node, false, gene2_innovation, 0);

            // store innovation
            var innovation = try Innovation.init_for_node(allocator, in_node.id, out_node.id, gene1_innovation, gene2_innovation, node.?.id, gene.?.innovation_num);
            try innovations.store_innovation(innovation);
        } else if (node != null and try self.has_node(node.?)) {
            // The same add node innovation occurred in the same genome (parent) - just skip.
            // This may happen when parent of this organism experienced the same mutation in current epoch earlier
            // and after that parent's genome was duplicated to child by mating and the same mutation parameters
            // was selected again (in_node.Id, out_node.Id, gene.InnovationNum). As result the innovation with given
            // parameters will be found and new node will be created with ID which already exists in child genome.
            // If proceed than we will have duplicated Node and genes - so we're skipping this.
            logger.info("GENOME: Add node innovation found [{any}] in the same genome [{d}] for node [{d}]\n{any}\n", .{ innovation_found, self.id, node.?.id, self }, @src());
            return false;
        }

        // now add the new NNode and new Genes to the Genome
        if (node != null and gene1 != null and gene2 != null) {
            // modifying Genome's gene/node list; use Genome's allocator
            self.genes = try self.gene_insert(self.allocator, self.genes, gene1);
            self.genes = try self.gene_insert(self.allocator, self.genes, gene2);
            self.nodes = try self.node_insert(self.allocator, self.nodes, node);
            return true;
        }
        // failed to create node or connecting genes
        return false;
    }

    pub fn mutate_link_weights(self: *Genome, rand: std.rand.Random, power: f64, rate: f64, mutation_type: MutatorType) !bool {
        if (self.genes.len == 0) {
            logger.err("Genome has no genes", .{}, @src());
            return GenomeError.GenomeHasNoGenes;
        }

        // occassionally, really mix things up
        var severe = false;
        if (rand.float(f64) > 0.5) {
            severe = true;
        }

        // walk through genes, perturbing their link's weights
        var num: f64 = 0.0;
        var genes_count: f64 = @as(f64, @floatFromInt(self.genes.len));
        var end_part = genes_count * 0.8;
        var gauss_point: f64 = undefined;
        var cold_gauss_point: f64 = undefined;
        for (self.genes) |gene| {
            // The following if determines the probabilities of doing cold gaussian
            // mutation, meaning the probability of replacing a link weight with
            // another, entirely random weight. It is meant to bias such mutations
            // to the tail of a genome, because that is where less time-tested genes
            // reside. The gauss_point and cold_gauss_point represent values above
            // which a random float will signify that kind of mutation.
            if (severe) {
                gauss_point = 0.3;
                cold_gauss_point = 0.1;
            } else if (genes_count >= 10.0 and num > end_part) {
                gauss_point = 0.5; // Mutate by modification % of connections
                cold_gauss_point = 0.3; // Mutate the rest by replacement % of the time
            } else {
                // 50% of the time, don't do any cold mutation
                gauss_point = 1.0 - rate;
                if (rand.float(f64) > 0.5) {
                    cold_gauss_point = gauss_point - 0.1;
                } else {
                    cold_gauss_point = gauss_point; // no cold mutation possible (see later)
                }
            }

            var random = @as(f64, @floatFromInt(math.rand_sign(i32, rand))) * rand.float(f64) * power;
            if (mutation_type == MutatorType.GaussianMutator) {
                var rand_choice = rand.float(f64);
                if (rand_choice > gauss_point) {
                    gene.link.cxn_weight += random;
                } else if (rand_choice > cold_gauss_point) {
                    gene.link.cxn_weight = random;
                }
            } else if (mutation_type == MutatorType.GoldGaussianMutator) {
                gene.link.cxn_weight = random;
            }

            // record the innovation
            gene.mutation_num = gene.link.cxn_weight;
            num += 1.0;
        }
        return true;
    }

    pub fn mutate_random_trait(self: *Genome, rand: std.rand.Random, ctx: *Options) !bool {
        if (self.traits.len == 0) {
            logger.err("Genome has no traits", .{}, @src());
            return GenomeError.GenomeHasNoTraits;
        }

        // choose random trait idx
        var trait_num = rand.uintLessThan(usize, self.traits.len);
        // retrieve and mutate trait
        self.traits[trait_num].mutate(rand, ctx.trait_mut_power, ctx.trait_param_mut_prob);
        return true;
    }

    pub fn mutate_link_trait(self: *Genome, rand: std.rand.Random, times: usize) !bool {
        if (self.traits.len == 0 or self.genes.len == 0) {
            logger.err("Genome has either no traits or genes", .{}, @src());
            return GenomeError.GenomeHasNoTraitsOrGenes;
        }

        var i: usize = 0;
        while (i < times) : (i += 1) {
            // choose random trait number
            var trait_num = rand.uintLessThan(usize, self.traits.len);

            // choose random link number
            var gene_num = rand.uintLessThan(usize, self.genes.len);

            // set the link to point to the new trait
            self.genes[gene_num].link.trait = self.traits[trait_num];
        }
        return true;
    }

    pub fn mutate_node_trait(self: *Genome, rand: std.rand.Random, times: usize) !bool {
        if (self.traits.len == 0 or self.nodes.len == 0) {
            logger.err("Genome has either no traits or nodes", .{}, @src());
            return GenomeError.GenomeHasNoTraitsOrNodes;
        }

        var i: usize = 0;
        while (i < times) : (i += 1) {
            // choose random trait
            var trait_num = rand.uintLessThan(usize, self.traits.len);

            // choose random node
            var node_num = rand.uintLessThan(usize, self.nodes.len);

            // set the node to point to the new trait
            self.nodes[node_num].trait = self.traits[trait_num];
        }
        return true;
    }

    pub fn mutate_toggle_enable(self: *Genome, rand: std.rand.Random, times: usize) !bool {
        if (self.genes.len == 0) {
            logger.err("Genome has no genes to toggle", .{}, @src());
            return GenomeError.GenomeHasNoGenes;
        }

        var i: usize = 0;
        while (i < times) : (i += 1) {
            // choose random gene
            var gene_num = rand.uintLessThan(usize, self.genes.len);

            var gene = self.genes[gene_num];
            if (gene.is_enabled) {
                // verify that another gene connects out of the in-node
                // to ensure that section of the network doesn't break
                // off and become isolated
                for (self.genes) |check_gene| {
                    if (check_gene.link.in_node.?.id == gene.link.in_node.?.id and check_gene.is_enabled and check_gene.innovation_num != gene.innovation_num) {
                        gene.is_enabled = false;
                        break;
                    }
                }
            }
        }
        return true;
    }

    /// finds first disabled gene and re-enables it
    pub fn mutate_gene_reenable(self: *Genome) !bool {
        if (self.genes.len == 0) {
            logger.err("Genome has no genes to re-enable", .{}, @src());
            return GenomeError.GenomeHasNoGenes;
        }
        for (self.genes) |gene| {
            if (!gene.is_enabled) {
                gene.is_enabled = true;
                break;
            }
        }
        return true;
    }

    pub fn mutate_all_nonstructural(self: *Genome, rand: std.rand.Random, ctx: *Options) !bool {
        var res = false;

        if (rand.float(f64) < ctx.mut_random_trait_prob) {
            // mutate random trait
            res = try self.mutate_random_trait(rand, ctx);
        }

        if (rand.float(f64) < ctx.mut_link_trait_prob) {
            // mutate link trait
            res = try self.mutate_link_trait(rand, 1);
        }

        if (rand.float(f64) < ctx.mut_node_trait_prob) {
            // mutate node trait
            res = try self.mutate_node_trait(rand, 1);
        }

        if (rand.float(f64) < ctx.mut_link_weights_prob) {
            // mutate link weight
            res = try self.mutate_link_weights(rand, ctx.weight_mut_power, 1.0, MutatorType.GaussianMutator);
        }

        if (rand.float(f64) < ctx.mut_toggle_enable_prob) {
            // mutate toggle enable
            res = try self.mutate_toggle_enable(rand, 1);
        }

        if (rand.float(f64) < ctx.mut_gene_reenable_prob) {
            // mutate gene re-enable
            res = try self.mutate_gene_reenable();
        }
        return res;
    }

    pub fn mate_multipoint(self: *Genome, allocator: std.mem.Allocator, rand: std.rand.Random, og: *Genome, genome_id: i64, fitness1: f64, fitness2: f64) !*Genome {
        // verify genomes have same trait count
        if (self.traits.len != og.traits.len) {
            logger.err("Genomes has different traits count, {d} != {d}", .{ self.traits.len, og.traits.len }, @src());
            return GenomeError.GenomesHaveDifferentTraitsCount;
        }

        // avg traits to create new traits for offspring
        var new_traits = try self.mate_traits(allocator, og);

        // new genes/nodes are created
        var new_genes = try allocator.alloc(*Gene, 0);
        var new_nodes = try allocator.alloc(*NNode, 0);
        var child_nodes_map = std.AutoHashMap(i64, *NNode).init(allocator);
        defer child_nodes_map.deinit();

        // ensure all sensors and outputs are included (in case some inputs are disconnected)
        for (og.nodes) |node| {
            if (node.neuron_type == NodeNeuronType.InputNeuron or node.neuron_type == NodeNeuronType.BiasNeuron or node.neuron_type == NodeNeuronType.OutputNeuron) {
                var node_trait_num: usize = 0;
                if (node.trait != null) {
                    node_trait_num = @as(usize, @intCast(node.trait.?.id.? - self.traits[0].id.?));
                }

                // create a new node off of the sensor or output
                var o_node = try NNode.init_copy(allocator, node, new_traits[node_trait_num]);

                // add the node
                new_nodes = try self.node_insert(allocator, new_nodes, o_node);
                try child_nodes_map.put(o_node.id, o_node);
            }
        }

        // determine which genome is better
        var p1_better = false;
        if (fitness1 > fitness2 or (fitness1 == fitness2 and self.genes.len < og.genes.len)) {
            p1_better = true;
        }

        // iterate through genes of both parents
        var i_1: usize = 0;
        var i_2: usize = 0;
        var size1 = self.genes.len;
        var size2 = og.genes.len;
        var chosen_gene: *Gene = undefined;
        while (i_1 < size1 or i_2 < size2) {
            var skip = false;
            var disable = false;

            // choose the best gene
            if (i_1 >= size1) {
                chosen_gene = og.genes[i_2];
                i_2 += 1;
                if (p1_better) {
                    skip = true; // skip excess from the worse genome
                }
            } else if (i_2 >= size2) {
                chosen_gene = self.genes[i_1];
                i_1 += 1;
                if (!p1_better) {
                    skip = true; // skip excess from the worse genome
                }
            } else {
                var p1_gene = self.genes[i_1];
                var p2_gene = og.genes[i_2];

                // extract current innovation numbers
                var p1_innov = p1_gene.innovation_num;
                var p2_innov = p2_gene.innovation_num;

                if (p1_innov == p2_innov) {
                    if (rand.float(f64) < 0.5) {
                        chosen_gene = p1_gene;
                    } else {
                        chosen_gene = p2_gene;
                    }

                    // if one is disabled, the corresponding gene in the offspring will likely be disabled
                    if (!p1_gene.is_enabled or !p2_gene.is_enabled and rand.float(f64) < 0.75) {
                        disable = true;
                    }
                    i_1 += 1;
                    i_2 += 1;
                } else if (p1_innov < p2_innov) {
                    chosen_gene = p1_gene;
                    i_1 += 1;
                    if (!p1_better) {
                        skip = true; // skip disjoint from the worse genome
                    }
                } else {
                    chosen_gene = p2_gene;
                    i_2 += 1;
                    if (p1_better) {
                        skip = true; // skip disjoint from the worse genome
                    }
                }
            }

            // Uncomment this line to let growth go faster (from both parents excesses)
            // skip = false;

            // check whether chosen gene conflicts with an already
            // chosen gene; i.e. do they represent the same link
            if (!skip) {
                for (new_genes) |gene| {
                    if (gene.link.is_genetically_eql(chosen_gene.link)) {
                        skip = true;
                        break;
                    }
                }
            }

            // add the chosen gene to the offspring
            if (!skip) {
                // check for the nodes, add if not already in new Genome offspring
                var in_node = chosen_gene.link.in_node.?;
                var out_node = chosen_gene.link.out_node.?;

                // check for in_node's existence
                var new_in_node: ?*NNode = null;
                for (new_nodes) |node| {
                    if (node.id == in_node.id) {
                        new_in_node = node;
                        break;
                    }
                }

                if (new_in_node == null) {
                    // node doesn't exist in new Genome, add it's normalized trait
                    // num for new NNode
                    var in_node_trait_num: usize = 0;
                    if (in_node.trait != null) {
                        in_node_trait_num = @as(usize, @intCast(in_node.trait.?.id.? - self.traits[0].id.?));
                    }
                    new_in_node = try NNode.init_copy(allocator, in_node, new_traits[in_node_trait_num]);
                    new_nodes = try self.node_insert(allocator, new_nodes, new_in_node);
                    try child_nodes_map.put(new_in_node.?.id, new_in_node.?);
                }

                // check for out_node's existence
                var new_out_node: ?*NNode = null;
                for (new_nodes) |node| {
                    if (node.id == out_node.id) {
                        new_out_node = node;
                        break;
                    }
                }

                if (new_out_node == null) {
                    // node doesn't exist in new Genome, add it's normalized trait
                    // num for new NNode
                    var out_node_trait_num: usize = 0;
                    if (out_node.trait != null) {
                        out_node_trait_num = @as(usize, @intCast(out_node.trait.?.id.? - self.traits[0].id.?));
                    }
                    new_out_node = try NNode.init_copy(allocator, out_node, new_traits[out_node_trait_num]);
                    new_nodes = try self.node_insert(allocator, new_nodes, new_out_node);
                    try child_nodes_map.put(new_out_node.?.id, new_out_node.?);
                }

                // add the gene
                var gene_trait_num: usize = 0;
                if (chosen_gene.link.trait != null) {
                    // The subtracted number normalizes depending on whether traits start counting at 1 or 0
                    gene_trait_num = @as(usize, @intCast(chosen_gene.link.trait.?.id.? - self.traits[0].id.?));
                }

                var gene = try Gene.init_copy(allocator, chosen_gene, new_traits[gene_trait_num], new_in_node, new_out_node);
                if (disable) {
                    gene.is_enabled = false;
                }
                new_genes = try self.gene_insert(allocator, new_genes, gene);
            } // end skip
        } // end for

        // check if parent's MIMO control genes should be inherited
        if ((self.control_genes != null and self.control_genes.?.len != 0) or (og.control_genes != null and og.control_genes.?.len != 0)) {
            // MIMO control genes found at least in one parent - append it to child if appropriate
            var control_data = try self.mate_modules(allocator, &child_nodes_map, og);
            defer control_data.deinit();
            if (control_data.nodes.len > 0) {
                var tmp_nodes = std.ArrayList(*NNode).fromOwnedSlice(allocator, new_nodes);
                try tmp_nodes.appendSlice(control_data.nodes);
                new_nodes = try tmp_nodes.toOwnedSlice();
            }
            defer allocator.free(control_data.nodes);
            // return new modular genome
            return Genome.init_modular(allocator, genome_id, new_traits, new_nodes, new_genes, control_data.modules);
        }
        // return plain new Genome
        return Genome.init(allocator, genome_id, new_traits, new_nodes, new_genes);
    }

    pub fn mate_multipoint_avg(self: *Genome, allocator: std.mem.Allocator, rand: std.rand.Random, og: *Genome, genome_id: i64, fitness1: f64, fitness2: f64) !*Genome {
        // verify both Genomes have same number of traits
        if (self.traits.len != og.traits.len) {
            logger.err("Genomes has different traits count, {d} != {d}", .{ self.traits.len, og.traits.len }, @src());
            return GenomeError.GenomesHaveDifferentTraitsCount;
        }

        // avg traits to create new traits for offspring
        var new_traits = try self.mate_traits(allocator, og);

        // new genes/nodes are created
        var new_genes = try allocator.alloc(*Gene, 0);
        var new_nodes = try allocator.alloc(*NNode, 0);
        var child_nodes_map = std.AutoHashMap(i64, *NNode).init(allocator);
        defer child_nodes_map.deinit();

        for (og.nodes) |node| {
            if (node.neuron_type == NodeNeuronType.InputNeuron or node.neuron_type == NodeNeuronType.BiasNeuron or node.neuron_type == NodeNeuronType.OutputNeuron) {
                var node_trait_num: usize = 0;
                if (node.trait != null) {
                    node_trait_num = @as(usize, @intCast(node.trait.?.id.? - self.traits[0].id.?));
                }
                // create a new node off the sensor or output
                var new_node = try NNode.init_copy(allocator, node, new_traits[node_trait_num]);
                // add the new node
                new_nodes = try self.node_insert(allocator, new_nodes, new_node);
                try child_nodes_map.put(new_node.id, new_node);
            }
        }

        // Determine which genome is better.
        // The worse genome should not be allowed to add extra structural baggage.
        // If they are the same, use the smaller one's disjoint and excess genes only.
        var p1_better = false; // indicates whether the first genome is better or not
        if (fitness1 > fitness2 or (fitness1 == fitness2 and self.genes.len < og.genes.len)) {
            p1_better = true;
        }

        // init avg_gene - this gene is used to hold the avg of the two genes being bred
        var avg_gene = try Gene.init_with_trait(allocator, null, 0.0, null, null, false, 0, 0.0);
        defer avg_gene.deinit();

        var i_1: usize = 0;
        var i_2: usize = 0;
        var size1 = self.genes.len;
        var size2 = og.genes.len;
        var chosen_gene: *Gene = undefined;
        while (i_1 < size1 or i_2 < size2) {
            var skip = false;
            avg_gene.is_enabled = true; // default to enabled

            // choose the best gene
            if (i_1 >= size1) {
                chosen_gene = og.genes[i_2];
                i_2 += 1;
                if (p1_better) {
                    skip = true; // skip excess from the worse genome
                }
            } else if (i_2 >= size2) {
                chosen_gene = self.genes[i_1];
                i_1 += 1;
                if (!p1_better) {
                    skip = true; // skip excess from the worse genome
                }
            } else {
                var p1_gene = self.genes[i_1];
                var p2_gene = og.genes[i_2];

                // extract current innovation numbers
                var p1_innov = p1_gene.innovation_num;
                var p2_innov = p2_gene.innovation_num;

                if (p1_innov == p2_innov) {
                    // average them into the avg_gene
                    if (rand.float(f64) > 0.5) {
                        avg_gene.link.trait = p1_gene.link.trait;
                    } else {
                        avg_gene.link.trait = p2_gene.link.trait;
                    }
                    avg_gene.link.cxn_weight = (p1_gene.link.cxn_weight + p2_gene.link.cxn_weight) / 2;

                    if (rand.float(f64) > 0.5) {
                        avg_gene.link.in_node = p1_gene.link.in_node;
                    } else {
                        avg_gene.link.in_node = p2_gene.link.in_node;
                    }
                    if (rand.float(f64) > 0.5) {
                        avg_gene.link.out_node = p1_gene.link.out_node;
                    } else {
                        avg_gene.link.out_node = p2_gene.link.out_node;
                    }
                    if (rand.float(f64) > 0.5) {
                        avg_gene.link.is_recurrent = p1_gene.link.is_recurrent;
                    } else {
                        avg_gene.link.is_recurrent = p2_gene.link.is_recurrent;
                    }

                    avg_gene.innovation_num = p1_innov;
                    avg_gene.mutation_num = (p1_gene.mutation_num + p2_gene.mutation_num) / 2;
                    if (!p1_gene.is_enabled or !p2_gene.is_enabled and rand.float(f64) < 0.75) {
                        avg_gene.is_enabled = false;
                    }

                    chosen_gene = avg_gene;
                    i_1 += 1;
                    i_2 += 1;
                } else if (p1_innov < p2_innov) {
                    chosen_gene = p1_gene;
                    i_1 += 1;
                    if (!p1_better) {
                        skip = true; // skip disjoint from the worse genome
                    }
                } else {
                    chosen_gene = p2_gene;
                    i_2 += 1;
                    if (p1_better) {
                        skip = true; // skip disjoint from the worse genome
                    }
                }
            }

            // Uncomment this line to let growth go faster (from both parents excesses)
            // skip = false;

            // check whether chosen gene conflicts with an already chosen gene (i.e. do they represent the same link)
            if (!skip) {
                for (new_genes) |gene| {
                    if (gene.link.is_genetically_eql(chosen_gene.link)) {
                        skip = true;
                        break;
                    }
                }
            }

            if (!skip) {
                // add the chosen gene to the offspring

                // check for the nodes, add them if not in offspring yet
                var in_node = chosen_gene.link.in_node.?;
                var out_node = chosen_gene.link.out_node.?;

                // check for in_node's existence
                var new_in_node: ?*NNode = null;
                for (new_nodes) |node| {
                    if (node.id == in_node.id) {
                        new_in_node = node;
                        break;
                    }
                }

                if (new_in_node == null) {
                    // node doesn't exist, so we have to add its normalized trait
                    // number for new NNode
                    var in_node_trait_num: usize = 0;
                    if (in_node.trait != null) {
                        in_node_trait_num = @as(usize, @intCast(in_node.trait.?.id.? - self.traits[0].id.?));
                    }
                    new_in_node = try NNode.init_copy(allocator, in_node, new_traits[in_node_trait_num]);
                    new_nodes = try self.node_insert(allocator, new_nodes, new_in_node);
                    try child_nodes_map.put(new_in_node.?.id, new_in_node.?);
                }

                // check for out_node's existence
                var new_out_node: ?*NNode = null;
                for (new_nodes) |node| {
                    if (node.id == out_node.id) {
                        new_out_node = node;
                        break;
                    }
                }

                if (new_out_node == null) {
                    // node doesn't exist, so we have to add its normalized trait
                    // number for new NNode
                    var out_node_trait_num: usize = 0;
                    if (out_node.trait != null) {
                        out_node_trait_num = @as(usize, @intCast(out_node.trait.?.id.? - self.traits[0].id.?));
                    }
                    new_out_node = try NNode.init_copy(allocator, out_node, new_traits[out_node_trait_num]);
                    new_nodes = try self.node_insert(allocator, new_nodes, new_out_node);
                    try child_nodes_map.put(new_out_node.?.id, new_out_node.?);
                }

                // add the gene
                var gene_trait_num: usize = 0;
                if (chosen_gene.link.trait != null) {
                    // the subtracted number normalizes depending on whether traits start counting at 1 or 0
                    gene_trait_num = @as(usize, @intCast(chosen_gene.link.trait.?.id.? - self.traits[0].id.?));
                }
                var gene = try Gene.init_copy(allocator, chosen_gene, new_traits[gene_trait_num], new_in_node, new_out_node);
                new_genes = try self.gene_insert(allocator, new_genes, gene);
            } // end skip
        } // end for

        // check if parent's MIMO control genes should be inherited
        if ((self.control_genes != null and self.control_genes.?.len != 0) or (og.control_genes != null and og.control_genes.?.len != 0)) {
            // MIMO control genes found at least in one parent - append it to child if appropriate
            var control_data = try self.mate_modules(allocator, &child_nodes_map, og);
            defer control_data.deinit();
            if (control_data.nodes.len > 0) {
                var tmp_nodes = std.ArrayList(*NNode).init(allocator);
                try tmp_nodes.appendSlice(new_nodes);
                try tmp_nodes.appendSlice(control_data.nodes);
                new_nodes = try tmp_nodes.toOwnedSlice();
            }
            defer allocator.free(control_data.nodes);
            // return new modular genome
            return Genome.init_modular(allocator, genome_id, new_traits, new_nodes, new_genes, control_data.modules);
        }
        return Genome.init(allocator, genome_id, new_traits, new_nodes, new_genes);
    }

    pub fn mate_singlepoint(self: *Genome, allocator: std.mem.Allocator, rand: std.rand.Random, og: *Genome, genome_id: i64) !*Genome {
        // verify both Genomes have same number of traits
        if (self.traits.len != og.traits.len) {
            logger.err("Genomes has different traits count, {d} != {d}", .{ self.traits.len, og.traits.len }, @src());
            return GenomeError.GenomesHaveDifferentTraitsCount;
        }

        // avg traits to create new traits for offspring
        var new_traits = try self.mate_traits(allocator, og);

        // new genes/nodes are created
        var new_genes = try allocator.alloc(*Gene, 0);
        var new_nodes = try allocator.alloc(*NNode, 0);
        var child_nodes_map = std.AutoHashMap(i64, *NNode).init(allocator);
        defer child_nodes_map.deinit();

        // make sure all sensors and outputs are included (in case some inputs are disconnected)
        for (og.nodes) |node| {
            if (node.neuron_type == NodeNeuronType.InputNeuron or node.neuron_type == NodeNeuronType.BiasNeuron or node.neuron_type == NodeNeuronType.OutputNeuron) {
                var node_trait_num: usize = 0;
                if (node.trait != null) {
                    node_trait_num = @as(usize, @intCast(node.trait.?.id.? - self.traits[0].id.?));
                }
                // create a enw node off the sensor/output
                var new_node = try NNode.init_copy(allocator, node, new_traits[node_trait_num]);
                // add the new node
                new_nodes = try self.node_insert(allocator, new_nodes, new_node);
                try child_nodes_map.put(new_node.id, new_node);
            }
        }

        // setup avg_gene - used to hold the avg of the 2 genes being bred
        var avg_gene = try Gene.init_with_trait(allocator, null, 0.0, null, null, false, 0, 0.0);
        defer avg_gene.deinit();

        var p1_stop: usize = 0;
        var p2_stop: usize = 0;
        var stopper: usize = 0;
        var cross_point: usize = 0;
        var p1_genes: []*Gene = undefined;
        var p2_genes: []*Gene = undefined;
        var size1 = self.genes.len;
        var size2 = og.genes.len;
        if (size1 < size2) {
            cross_point = rand.uintLessThan(usize, size1);
            p1_stop = size1;
            p2_stop = size2;
            stopper = size2;
            p1_genes = self.genes;
            p2_genes = og.genes;
        } else {
            cross_point = rand.uintLessThan(usize, size2);
            p1_stop = size2;
            p2_stop = size1;
            stopper = size1;
            p1_genes = og.genes;
            p2_genes = self.genes;
        }

        var chosen_gene: ?*Gene = null;
        var gene_counter: usize = 0;
        var i_1: usize = 0;
        var i_2: usize = 0;
        // walk through the Genes of each parent until both Genomes end
        while (i_2 < stopper) {
            var skip = false;
            avg_gene.is_enabled = true; // default to enabled
            if (i_1 == p1_stop) {
                chosen_gene = p2_genes[i_2];
                i_2 += 1;
            } else if (i_2 == p2_stop) {
                chosen_gene = p1_genes[i_1];
                i_1 += 1;
            } else {
                var p1_gene = p1_genes[i_1];
                var p2_gene = p2_genes[i_2];

                // extract current innovation numbers
                var p1_innov = p1_gene.innovation_num;
                var p2_innov = p2_gene.innovation_num;

                if (p1_innov == p2_innov) {
                    // pick chosen gene based on whether we've crossed yet
                    if (gene_counter < cross_point) {
                        chosen_gene = p1_gene;
                    } else if (gene_counter > cross_point) {
                        chosen_gene = p2_gene;
                    } else {
                        // we are at cross_point here - avg genes into `avg_gene`
                        if (rand.float(f64) > 0.5) {
                            avg_gene.link.trait = p1_gene.link.trait;
                        } else {
                            avg_gene.link.trait = p2_gene.link.trait;
                        }
                        // average weights
                        avg_gene.link.cxn_weight = (p1_gene.link.cxn_weight + p2_gene.link.cxn_weight) / 2;

                        if (rand.float(f64) > 0.5) {
                            avg_gene.link.in_node = p1_gene.link.in_node;
                        } else {
                            avg_gene.link.in_node = p2_gene.link.in_node;
                        }
                        if (rand.float(f64) > 0.5) {
                            avg_gene.link.out_node = p1_gene.link.out_node;
                        } else {
                            avg_gene.link.out_node = p2_gene.link.out_node;
                        }
                        if (rand.float(f64) > 0.5) {
                            avg_gene.link.is_recurrent = p1_gene.link.is_recurrent;
                        } else {
                            avg_gene.link.is_recurrent = p2_gene.link.is_recurrent;
                        }

                        avg_gene.innovation_num = p1_innov;
                        avg_gene.mutation_num = (p1_gene.mutation_num + p2_gene.mutation_num) / 2;
                        if (!p1_gene.is_enabled or !p2_gene.is_enabled and rand.float(f64) < 0.75) {
                            avg_gene.is_enabled = false;
                        }
                        chosen_gene = avg_gene;
                    }
                    i_1 += 1;
                    i_2 += 1;
                    gene_counter += 1;
                } else if (p1_innov < p2_innov) {
                    if (gene_counter < cross_point) {
                        chosen_gene = p1_gene;
                        i_1 += 1;
                        gene_counter += 1;
                    } else {
                        chosen_gene = p2_gene;
                        i_2 += 1;
                    }
                } else {
                    // p2_innov < p1_innov
                    i_2 += 1;
                    // special case: we need to skip to the next iteration
                    // because this Gene is before the cross_point on the wrong Genome
                    skip = true;
                }
            }
            if (chosen_gene == null) {
                // no gene was chosen - no need to process further - exit cycle
                break;
            }

            // check whether chosen gene conflicts w already chosen gene (i.e. do they represent the same link)
            if (!skip) {
                for (new_genes) |gene| {
                    if (gene.link.is_genetically_eql(chosen_gene.?.link)) {
                        skip = true;
                        break;
                    }
                }
            }

            // add chosen_gene to the offspring
            if (!skip) {
                // check for the nodes, add if not in offspring yet
                var in_node = chosen_gene.?.link.in_node.?;
                var out_node = chosen_gene.?.link.out_node.?;

                // check for existence of `in_node`
                var new_in_node: ?*NNode = null;
                for (new_nodes) |node| {
                    if (node.id == in_node.id) {
                        new_in_node = node;
                        break;
                    }
                }

                if (new_in_node == null) {
                    // Here we know the node doesn't exist so we have to add it normalized trait
                    // number for new NNode
                    var in_node_trait_num: usize = 0;
                    if (in_node.trait != null) {
                        in_node_trait_num = @as(usize, @intCast(in_node.trait.?.id.? - self.traits[0].id.?));
                    }
                    new_in_node = try NNode.init_copy(allocator, in_node, new_traits[in_node_trait_num]);
                    new_nodes = try self.node_insert(allocator, new_nodes, new_in_node);
                    try child_nodes_map.put(new_in_node.?.id, new_in_node.?);
                }

                // check for existence of `out_node`
                var new_out_node: ?*NNode = null;
                for (new_nodes) |node| {
                    if (node.id == out_node.id) {
                        new_out_node = node;
                        break;
                    }
                }
                if (new_out_node == null) {
                    // Here we know the node doesn't exist so we have to add it normalized trait
                    // number for new NNode
                    var out_node_trait_num: usize = 0;
                    if (out_node.trait != null) {
                        out_node_trait_num = @as(usize, @intCast(out_node.trait.?.id.? - self.traits[0].id.?));
                    }
                    new_out_node = try NNode.init_copy(allocator, out_node, new_traits[out_node_trait_num]);
                    new_nodes = try self.node_insert(allocator, new_nodes, new_out_node);
                    try child_nodes_map.put(new_out_node.?.id, new_out_node.?);
                }

                // add the gene
                var gene_trait_num: usize = 0;
                if (chosen_gene.?.link.trait != null) {
                    gene_trait_num = @as(usize, @intCast(chosen_gene.?.link.trait.?.id.? - self.traits[0].id.?));
                }
                var gene = try Gene.init_copy(allocator, chosen_gene.?, new_traits[gene_trait_num], new_in_node, new_out_node);
                new_genes = try self.gene_insert(allocator, new_genes, gene);
            } // end skip
        } // end for

        // check whether parent's MIMO control genes should be inherited
        if ((self.control_genes != null and self.control_genes.?.len != 0) or (og.control_genes != null and og.control_genes.?.len != 0)) {
            // MIMO control genes found at least in one parent - append it to child if appropriate
            var control_data = try self.mate_modules(allocator, &child_nodes_map, og);
            defer control_data.deinit();
            if (control_data.nodes.len > 0) {
                var tmp_nodes = std.ArrayList(*NNode).fromOwnedSlice(allocator, new_nodes);
                try tmp_nodes.appendSlice(control_data.nodes);
                new_nodes = try tmp_nodes.toOwnedSlice();
            }
            defer allocator.free(control_data.nodes);
            // return new modular genome
            return Genome.init_modular(allocator, genome_id, new_traits, new_nodes, new_genes, control_data.modules);
        }
        return Genome.init(allocator, genome_id, new_traits, new_nodes, new_genes);
    }

    fn mate_modules(self: *Genome, allocator: std.mem.Allocator, child_nodes: *std.AutoHashMap(i64, *NNode), og: *Genome) !*ModuleMate {
        var res = try ModuleMate.init(allocator);

        var parent_modules = std.ArrayList(*MIMOControlGene).init(allocator);
        var extra_nodes = std.ArrayList(*NNode).init(allocator);
        if (self.control_genes != null) {
            var current_genome_modules = try self.find_modules_intersection(allocator, child_nodes, self.control_genes.?);
            defer allocator.free(current_genome_modules);

            if (current_genome_modules.len > 0) {
                try parent_modules.appendSlice(current_genome_modules);
            }
        }
        if (og.control_genes != null) {
            var out_genome_modules = try self.find_modules_intersection(allocator, child_nodes, og.control_genes.?);
            defer allocator.free(out_genome_modules);
            if (out_genome_modules.len > 0) {
                try parent_modules.appendSlice(out_genome_modules);
            }
        }

        if (parent_modules.items.len == 0) {
            res.nodes = try extra_nodes.toOwnedSlice();
            res.modules = try parent_modules.toOwnedSlice();
            return res;
        }

        for (parent_modules.items) |cg| {
            for (cg.io_nodes) |n| {
                if (!child_nodes.contains(n.id)) {
                    try extra_nodes.append(n);
                }
            }
        }

        res.nodes = try extra_nodes.toOwnedSlice();
        res.modules = try parent_modules.toOwnedSlice();
        return res;
    }

    fn find_modules_intersection(_: *Genome, allocator: std.mem.Allocator, nodes: *std.AutoHashMap(i64, *NNode), genes: []*MIMOControlGene) ![]*MIMOControlGene {
        var modules = std.ArrayList(*MIMOControlGene).init(allocator);
        for (genes) |cg| {
            if (cg.has_intersection(nodes)) {
                // attempt copying the MIMOControlGene
                var new_cg = try MIMOControlGene.init_from_copy(allocator, cg, try NNode.init_copy(allocator, cg.control_node, cg.control_node.trait));
                try modules.append(new_cg);
            }
        }
        return modules.toOwnedSlice();
    }

    pub fn mate_traits(self: *Genome, allocator: std.mem.Allocator, og: *Genome) ![]*Trait {
        var new_traits = try allocator.alloc(*Trait, self.traits.len);
        errdefer allocator.free(new_traits);
        for (self.traits, 0..) |tr, i| {
            new_traits[i] = try Trait.new_trait_avg(allocator, tr, og.traits[i]);
        }
        return new_traits;
    }

    pub fn write_to_file(self: *Genome, path: []const u8) !void {
        const dir_path = std.fs.path.dirname(path);
        const file_name = std.fs.path.basename(path);
        var file_dir: std.fs.Dir = undefined;
        if (dir_path != null) {
            file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
        } else {
            file_dir = std.fs.cwd();
        }
        var output_file = try file_dir.createFile(file_name, .{});
        defer output_file.close();

        // marks the start of genome encoding written to file
        try output_file.writer().print("genomestart {d}\n", .{self.id});

        // write traits
        for (self.traits) |tr| {
            try output_file.writer().print("trait {d} ", .{tr.id.?});
            for (tr.params, 0..) |p, i| {
                if (i < tr.params.len - 1) {
                    try output_file.writer().print("{d} ", .{p});
                } else {
                    try output_file.writer().print("{d}\n", .{p});
                }
            }
        }

        // write nodes
        for (self.nodes) |nd| {
            _ = try output_file.write("node ");
            var trait_id: i64 = 0;
            if (nd.trait != null) {
                trait_id = nd.trait.?.id.?;
            }
            var act_str = nd.activation_type.activation_name_by_type();
            try output_file.writer().print("{d} {d} {d} {d} {s}\n", .{ nd.id, trait_id, @intFromEnum(nd.node_type()), @intFromEnum(nd.neuron_type), act_str });
        }

        // write genes
        for (self.genes) |gn| {
            _ = try output_file.write("gene ");
            var link = gn.link;
            var trait_id: i64 = 0;
            if (link.trait != null) {
                trait_id = link.trait.?.id.?;
            }
            var in_node_id = link.in_node.?.id;
            var out_node_id = link.out_node.?.id;
            var weight = link.cxn_weight;
            var recurrent = link.is_recurrent;
            var innov_num = gn.innovation_num;
            var mut_num = gn.mutation_num;
            var enabled = gn.is_enabled;
            try output_file.writer().print("{d} {d} {d} {d} {any} {d} {d} {any}\n", .{ trait_id, in_node_id, out_node_id, weight, recurrent, innov_num, mut_num, enabled });
        }

        // marks the end of genome encoding written to file
        try output_file.writer().print("genomeend {d}", .{self.id});
    }

    pub fn read_from_file(allocator: std.mem.Allocator, path: []const u8) !*Genome {
        const dir_path = std.fs.path.dirname(path);
        const file_name = std.fs.path.basename(path);
        var file_dir: std.fs.Dir = undefined;
        if (dir_path != null) {
            file_dir = try std.fs.cwd().makeOpenPath(dir_path.?, .{});
        } else {
            file_dir = std.fs.cwd();
        }
        var file = try file_dir.openFile(file_name, .{});
        const file_size = (try file.stat()).size;
        var buf = try allocator.alloc(u8, file_size);
        defer allocator.free(buf);
        try file.reader().readNoEof(buf);
        var genome_id: i64 = undefined;
        var trait_list = std.ArrayList(*Trait).init(allocator);
        var node_list = std.ArrayList(*NNode).init(allocator);
        var gene_list = std.ArrayList(*Gene).init(allocator);
        var new_line_iterator = std.mem.split(u8, buf, "\n");
        while (new_line_iterator.next()) |line| {
            var split = std.mem.split(u8, line, " ");
            var first = split.first();
            var rest = split.rest();
            if (split.next() == null) {
                std.debug.print("line: [{s}] can not be split when reading Genome", .{line});
                return error.MalformedGenomeFile;
            }

            // parse traits
            if (std.mem.eql(u8, first, "trait")) {
                var new_trait = try Trait.read_from_file(allocator, rest);
                try trait_list.append(new_trait);
            }

            // parse nodes
            if (std.mem.eql(u8, first, "node")) {
                var new_node = try NNode.read_from_file(allocator, rest, trait_list.items);
                try node_list.append(new_node);
            }

            // parse genes
            if (std.mem.eql(u8, first, "gene")) {
                var new_gene = try Gene.read_from_file(allocator, rest, trait_list.items, node_list.items);
                try gene_list.append(new_gene);
            }

            // parse genome id
            if (std.mem.eql(u8, first, "genomeend")) {
                genome_id = try std.fmt.parseInt(i64, rest, 10);
            }
        }
        return Genome.init(allocator, genome_id, try trait_list.toOwnedSlice(), try node_list.toOwnedSlice(), try gene_list.toOwnedSlice());
    }
};

pub const ModuleMate = struct {
    allocator: std.mem.Allocator,
    nodes: []*NNode,
    modules: []*MIMOControlGene,

    pub fn init(allocator: std.mem.Allocator) !*ModuleMate {
        var self = try allocator.create(ModuleMate);
        self.* = .{
            .allocator = allocator,
            .nodes = undefined,
            .modules = undefined,
        };
        return self;
    }

    pub fn deinit(self: *ModuleMate) void {
        self.allocator.destroy(self);
    }
};

// test helper funcs

pub fn build_test_genome(allocator: std.mem.Allocator, id: usize) !*Genome {
    // init traits
    var traits = try allocator.alloc(*Trait, 3);
    traits[0] = try Trait.init(allocator, 8);
    traits[0].id = 1;
    traits[1] = try Trait.init(allocator, 8);
    traits[1].id = 2;
    traits[2] = try Trait.init(allocator, 8);
    traits[2].id = 3;
    for (traits[0].params, 0..) |*p, i| {
        p.* = if (i == 0) 0.1 else 0;
        traits[1].params[i] = if (i == 0) 0.3 else 0;
        traits[2].params[i] = if (i == 0) 0.2 else 0;
    }

    // init nodes
    var nodes = try allocator.alloc(*NNode, 4);
    nodes[0] = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    nodes[0].activation_type = neat_math.NodeActivationType.NullActivation;
    nodes[1] = try NNode.init(allocator, 2, NodeNeuronType.InputNeuron);
    nodes[1].activation_type = neat_math.NodeActivationType.NullActivation;
    nodes[2] = try NNode.init(allocator, 3, NodeNeuronType.BiasNeuron);
    nodes[2].activation_type = neat_math.NodeActivationType.SigmoidSteepenedActivation;
    nodes[3] = try NNode.init(allocator, 4, NodeNeuronType.OutputNeuron);
    nodes[3].activation_type = neat_math.NodeActivationType.SigmoidSteepenedActivation;

    // init genes
    var genes = try allocator.alloc(*Gene, 3);
    genes[0] = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, traits[0], 1.5, nodes[0], nodes[3], false), 1, 0, true);
    genes[1] = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, traits[2], 2.5, nodes[1], nodes[3], false), 2, 0, true);
    genes[2] = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, traits[1], 3.5, nodes[2], nodes[3], false), 3, 0, true);
    return Genome.init(allocator, @as(i64, @intCast(id)), traits, nodes, genes);
}

pub fn build_test_modular_genome(allocator: std.mem.Allocator, id: usize) !*Genome {
    var genome = try build_test_genome(allocator, id);

    // append module with it's IO nodes
    var io_nodes = try allocator.alloc(*NNode, 3);
    defer allocator.free(io_nodes);
    io_nodes[0] = try NNode.init(allocator, 5, NodeNeuronType.HiddenNeuron);
    io_nodes[0].activation_type = neat_math.NodeActivationType.LinearActivation;
    io_nodes[1] = try NNode.init(allocator, 6, NodeNeuronType.HiddenNeuron);
    io_nodes[1].activation_type = neat_math.NodeActivationType.LinearActivation;
    io_nodes[2] = try NNode.init(allocator, 7, NodeNeuronType.HiddenNeuron);
    io_nodes[2].activation_type = neat_math.NodeActivationType.NullActivation;

    var tmp_nodes = std.ArrayList(*NNode).fromOwnedSlice(allocator, genome.nodes);
    try tmp_nodes.appendSlice(io_nodes);
    genome.nodes = try tmp_nodes.toOwnedSlice();

    // connect added nodes
    var io_cxn_genes = try allocator.alloc(*Gene, 3);
    defer allocator.free(io_cxn_genes);
    io_cxn_genes[0] = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, genome.traits[0], 1.5, genome.nodes[0], genome.nodes[4], false), 4, 0, true);
    io_cxn_genes[1] = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, genome.traits[2], 2.5, genome.nodes[1], genome.nodes[5], false), 5, 0, true);
    io_cxn_genes[2] = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, genome.traits[1], 3.5, genome.nodes[6], genome.nodes[3], false), 6, 0, true);
    var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, genome.genes);
    try tmp_genes.appendSlice(io_cxn_genes);
    genome.genes = try tmp_genes.toOwnedSlice();

    // add control gene
    var control_node = try NNode.init(allocator, 8, NodeNeuronType.HiddenNeuron);
    control_node.activation_type = neat_math.NodeActivationType.MultiplyModuleActivation;
    try control_node.incoming.append(try Link.init(allocator, 1.0, io_nodes[0], control_node, false));
    try control_node.incoming.append(try Link.init(allocator, 1.0, io_nodes[1], control_node, false));
    try control_node.outgoing.append(try Link.init(allocator, 1.0, control_node, io_nodes[2], false));
    genome.control_genes = try allocator.alloc(*MIMOControlGene, 1);
    genome.control_genes.?[0] = try MIMOControlGene.init(allocator, control_node, 7, 5.5, true);
    return genome;
}

test "Genome initialize random" {
    var allocator = std.testing.allocator;
    var new_id: i64 = 1;
    var in: i64 = 3;
    var out: i64 = 2;
    var n: i64 = 2;

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();
    var gnome = try Genome.init_rand(allocator, rand, new_id, in, out, n, 5, false, 0.5);
    defer gnome.deinit();
    try std.testing.expect(gnome.nodes.len == in + n + out);
    try std.testing.expect(gnome.genes.len >= in + n + out);
}

test "Genome genesis" {
    var allocator = std.testing.allocator;
    var gnome = try build_test_genome(allocator, 1);
    defer gnome.deinit();
    var net_id: i64 = 10;
    var net = try gnome.genesis(allocator, net_id);
    try std.testing.expect(net.id == net_id);
    try std.testing.expect(net.node_count() == @as(i64, @intCast(gnome.nodes.len)));
    try std.testing.expect(net.link_count() == @as(i64, @intCast(gnome.genes.len)));
}

test "Genome genesis modular" {
    var allocator = std.testing.allocator;
    var gnome = try build_test_modular_genome(allocator, 1);
    defer gnome.deinit();
    var net_id: i64 = 10;
    var net = try gnome.genesis(allocator, 10);
    try std.testing.expect(net.id == net_id);
    // check plain neuron nodes
    var neuron_nodes_count = @as(i64, @intCast(gnome.nodes.len + gnome.control_genes.?.len));
    try std.testing.expect(net.node_count() == neuron_nodes_count);
    // find extra nodes and links due to MIMO control genes
    var incoming: usize = 0;
    var outgoing: usize = 0;
    for (gnome.control_genes.?) |cg| {
        incoming += cg.control_node.incoming.items.len;
        outgoing += cg.control_node.outgoing.items.len;
    }
    var cxn_genes_count = @as(i64, @intCast(gnome.genes.len + incoming + outgoing));
    try std.testing.expect(net.link_count() == cxn_genes_count);
}

test "Genome duplicate" {
    var allocator = std.testing.allocator;
    var gnome = try build_test_genome(allocator, 1);
    defer gnome.deinit();
    var new_genome = try gnome.duplicate(allocator, 2);
    defer new_genome.deinit();
    try std.testing.expect(new_genome.id == 2);
    try std.testing.expect(new_genome.nodes.len == gnome.nodes.len);
    try std.testing.expect(new_genome.genes.len == gnome.genes.len);
    try std.testing.expect(new_genome.traits.len == gnome.traits.len);

    var equal = try gnome.is_equal(new_genome);
    try std.testing.expect(equal);
}

test "Genome duplicate modular" {
    var allocator = std.testing.allocator;
    var gnome = try build_test_modular_genome(allocator, 1);
    defer gnome.deinit();
    var new_gnome = try gnome.duplicate(allocator, 2);
    defer new_gnome.deinit();

    try std.testing.expect(new_gnome.id == 2);
    try std.testing.expect(gnome.traits.len == new_gnome.traits.len);
    try std.testing.expect(gnome.nodes.len == new_gnome.nodes.len);
    try std.testing.expect(gnome.genes.len == new_gnome.genes.len);
    try std.testing.expect(gnome.control_genes.?.len == new_gnome.control_genes.?.len);

    for (new_gnome.control_genes.?, gnome.control_genes.?) |cg, ocg| {
        // check incoming connection genes
        try std.testing.expect(cg.control_node.incoming.items.len == ocg.control_node.incoming.items.len);
        for (cg.control_node.incoming.items, ocg.control_node.incoming.items) |l, ol| {
            try std.testing.expect(l.is_genetically_eql(ol));
        }
        // check outgoing connection genes
        try std.testing.expect(cg.control_node.outgoing.items.len == ocg.control_node.outgoing.items.len);
        for (cg.control_node.outgoing.items, ocg.control_node.outgoing.items) |l, ol| {
            try std.testing.expect(l.is_genetically_eql(ol));
        }
    }
    try std.testing.expect(try gnome.is_equal(new_gnome));
}

test "Genome verify" {
    var allocator = std.testing.allocator;
    var gnome = try build_test_genome(allocator, 1);
    defer gnome.deinit();

    var res = try gnome.verify();
    try std.testing.expect(res);

    // check gene missing input node failure
    var new_in_node = try NNode.init(allocator, 100, NodeNeuronType.InputNeuron);
    defer new_in_node.deinit();
    var new_out_node = try NNode.init(allocator, 4, NodeNeuronType.OutputNeuron);
    defer new_out_node.deinit();
    var gene = try Gene.init(allocator, 1.0, new_in_node, new_out_node, false, 1, 1.0);
    var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome.genes);
    try tmp_genes.append(gene);
    gnome.genes = try tmp_genes.toOwnedSlice();
    res = gnome.verify() catch false;
    try std.testing.expect(!res);

    // check gene missing output node failure
    var gnome2 = try build_test_genome(allocator, 1);
    defer gnome2.deinit();
    var new_in_node2 = try NNode.init(allocator, 4, NodeNeuronType.InputNeuron);
    defer new_in_node2.deinit();
    var new_out_node2 = try NNode.init(allocator, 400, NodeNeuronType.OutputNeuron);
    defer new_out_node2.deinit();
    var gene2 = try Gene.init(allocator, 1.0, new_in_node2, new_out_node2, false, 1, 1.0);
    tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome2.genes);
    try tmp_genes.append(gene2);
    gnome2.genes = try tmp_genes.toOwnedSlice();
    res = gnome2.verify() catch false;
    try std.testing.expect(!res);

    // test duplicate genes failure
    var gnome3 = try build_test_genome(allocator, 1);
    defer gnome3.deinit();
    var new_in_node3 = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer new_in_node3.deinit();
    var new_out_node3 = try NNode.init(allocator, 1, NodeNeuronType.OutputNeuron);
    defer new_out_node3.deinit();
    var new_in_node4 = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer new_in_node4.deinit();
    var new_out_node4 = try NNode.init(allocator, 1, NodeNeuronType.OutputNeuron);
    defer new_out_node4.deinit();
    tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome3.genes);
    var gene3 = try Gene.init(allocator, 1.0, new_in_node3, new_out_node3, false, 1, 1.0);
    var gene4 = try Gene.init(allocator, 1.0, new_in_node3, new_out_node3, false, 1, 1.0);
    try tmp_genes.append(gene3);
    try tmp_genes.append(gene4);
    gnome3.genes = try tmp_genes.toOwnedSlice();
    res = gnome3.verify() catch false;
    try std.testing.expect(!res);
}

test "Genome gene insert" {
    var allocator = std.testing.allocator;
    var gnome = try build_test_genome(allocator, 1);
    defer gnome.deinit();
    var new_gene = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome.traits[2], 5.5, gnome.nodes[2], gnome.nodes[3], false), 5, 0, false);
    var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome.genes);
    try tmp_genes.append(new_gene);
    gnome.genes = try tmp_genes.toOwnedSlice();
    var new_gene2 = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome.traits[2], 5.5, gnome.nodes[2], gnome.nodes[3], false), 4, 0, false);
    gnome.genes = try gnome.gene_insert(allocator, gnome.genes, new_gene2);
    try std.testing.expect(gnome.genes.len == 5);
    for (gnome.genes, 0..) |gn, i| {
        try std.testing.expect(gn.innovation_num == i + 1);
    }
}

test "Genome compatability linear" {
    var allocator: std.mem.Allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try build_test_genome(allocator, 2);
    defer gnome2.deinit();

    // configuration
    var options = Options{
        .disjoint_coeff = 0.5,
        .excess_coeff = 0.5,
        .mut_diff_coeff = 0.5,
        .gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodLinear,
    };

    // test fully compatible
    var comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 0.0);

    // test incompatible
    var tmp_list = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome2.genes); // hacky means to append to slice
    var new_in_node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer new_in_node.deinit();
    var new_out_node = try NNode.init(allocator, 1, NodeNeuronType.OutputNeuron);
    defer new_out_node.deinit();
    var new_gene = try Gene.init(allocator, 1.0, new_in_node, new_out_node, false, 10, 1.0);
    try tmp_list.append(new_gene);
    gnome2.genes = try tmp_list.toOwnedSlice();
    comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 0.5);

    tmp_list = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome2.genes);
    var new_in_node1 = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer new_in_node1.deinit();
    var new_out_node1 = try NNode.init(allocator, 1, NodeNeuronType.OutputNeuron);
    defer new_out_node1.deinit();
    var new_gene2 = try Gene.init(allocator, 1.0, new_in_node1, new_out_node1, false, 5, 1.0);
    try tmp_list.append(new_gene2);
    gnome2.genes = try tmp_list.toOwnedSlice();
    comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 1.0);

    gnome2.genes[1].mutation_num = 6.0;
    comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 2.0);
}

test "Genome compatability fast" {
    var allocator: std.mem.Allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try build_test_genome(allocator, 2);
    defer gnome2.deinit();

    // configuration
    var options = Options{
        .disjoint_coeff = 0.5,
        .excess_coeff = 0.5,
        .mut_diff_coeff = 0.5,
        .gen_compat_method = GenomeCompatibilityMethod.GenomeCompatibilityMethodFast,
    };

    // test fully compatible
    var comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 0.0);

    // test incompatible
    var tmp_list = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome2.genes); // hacky means to append to slice
    var new_in_node = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer new_in_node.deinit();
    var new_out_node = try NNode.init(allocator, 1, NodeNeuronType.OutputNeuron);
    defer new_out_node.deinit();
    var new_gene = try Gene.init(allocator, 1.0, new_in_node, new_out_node, false, 10, 1.0);
    try tmp_list.append(new_gene);
    gnome2.genes = try tmp_list.toOwnedSlice();
    comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 0.5);

    tmp_list = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome2.genes);
    var new_in_node1 = try NNode.init(allocator, 1, NodeNeuronType.InputNeuron);
    defer new_in_node1.deinit();
    var new_out_node1 = try NNode.init(allocator, 1, NodeNeuronType.OutputNeuron);
    defer new_out_node1.deinit();
    var new_gene2 = try Gene.init(allocator, 1.0, new_in_node1, new_out_node1, false, 5, 1.0);
    try tmp_list.append(new_gene2);
    gnome2.genes = try tmp_list.toOwnedSlice();
    comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 1.0);

    gnome2.genes[1].mutation_num = 6.0;
    comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 2.0);
}

test "Genome compatability duplicate" {
    var allocator: std.mem.Allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try gnome1.duplicate(allocator, 2);
    defer gnome2.deinit();

    // configuration
    var options = Options{
        .disjoint_coeff = 0.5,
        .excess_coeff = 0.5,
        .mut_diff_coeff = 0.5,
    };

    var comp = gnome1.compatability(gnome2, &options);
    try std.testing.expect(comp == 0.0);
}

test "Genome mutate add link" {
    var allocator: std.mem.Allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    var prng = std.rand.DefaultPrng.init(3);
    const rand = prng.random();

    // configuration
    var options = Options{
        .recur_only_prob = 0.5,
        .new_link_tries = 10,
        .compat_threshold = 0.5,
        .pop_size = 1,
    };
    var pop = try Population.raw_init(allocator);
    defer pop.deinit();
    try pop.spawn(allocator, rand, gnome1, &options);

    // create gnome phenotype
    _ = try gnome1.genesis(allocator, 1);

    var res = try gnome1.mutate_add_link(allocator, rand, pop, &options);
    try std.testing.expect(res);

    // one gene was added innovNum = 3 + 1
    try std.testing.expect(pop.next_innov_number.loadUnchecked() == 4);
    try std.testing.expect(pop.innovations.items.len == 1);
    try std.testing.expect(gnome1.genes.len == 4);
    var gene = gnome1.genes[3];
    try std.testing.expect(gene.innovation_num == 4);
    try std.testing.expect(gene.link.is_recurrent);

    // add more NEURONS
    options.recur_only_prob = 0.0;
    var nodes = try allocator.alloc(*NNode, 2);
    defer allocator.free(nodes);
    nodes[0] = try NNode.init(allocator, 5, NodeNeuronType.HiddenNeuron);
    nodes[0].activation_type = neat_math.NodeActivationType.SigmoidSteepenedActivation;
    nodes[1] = try NNode.init(allocator, 6, NodeNeuronType.InputNeuron);
    nodes[1].activation_type = neat_math.NodeActivationType.SigmoidSteepenedActivation;

    var tmp_nodes = std.ArrayList(*NNode).fromOwnedSlice(allocator, gnome1.nodes);
    try tmp_nodes.appendSlice(nodes);
    gnome1.nodes = try tmp_nodes.toOwnedSlice();

    // do network genesis with new nodes added
    _ = try gnome1.genesis(allocator, 1);

    res = try gnome1.mutate_add_link(allocator, rand, pop, &options);
    try std.testing.expect(res);

    // one gene was added innovNum = 4 + 1
    try std.testing.expect(pop.next_innov_number.loadUnchecked() == 5);
    try std.testing.expect(pop.innovations.items.len == 2);
    try std.testing.expect(gnome1.genes.len == 5);

    gene = gnome1.genes[4];
    try std.testing.expect(gene.innovation_num == 5);
    // new gene must not be recurrent, because `options.recur_only_prob = 0.0`
    try std.testing.expect(gene.link.is_recurrent == false);
}

test "Genome mutate connect sensors" {
    var allocator = std.testing.allocator;

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    // test mutation with all inputs connected
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    // create gnome phenotype
    _ = try gnome1.genesis(allocator, 1);

    // configuration
    var options = Options{
        .pop_size = 1,
    };

    // init population with one organism
    var pop = try Population.raw_init(allocator);
    defer pop.deinit();
    try pop.spawn(allocator, rand, gnome1, &options);

    var res = try gnome1.mutate_connect_sensors(allocator, rand, pop, &options);
    try std.testing.expect(res == false); // all inputs are already connected; always returns false

    // test with disconnected input
    var new_node = try NNode.init(allocator, 5, NodeNeuronType.InputNeuron);
    new_node.activation_type = neat_math.NodeActivationType.SigmoidSteepenedActivation;
    var tmp_nodes = std.ArrayList(*NNode).fromOwnedSlice(allocator, gnome1.nodes);
    try tmp_nodes.append(new_node);
    gnome1.nodes = try tmp_nodes.toOwnedSlice();

    // create gnome phenotype
    _ = try gnome1.genesis(allocator, 1);
    res = try gnome1.mutate_connect_sensors(allocator, rand, pop, &options);
    try std.testing.expect(res); // new_node should be connected now
    try std.testing.expect(gnome1.genes.len == 4);
    try std.testing.expect(pop.innovations.items.len == 1);
    // one gene was added, expecting innovation + 1 (3+1)
    try std.testing.expect(pop.next_innov_number.loadUnchecked() == 4);
}

test "Genome mutate add node" {
    var allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    // create gnome phenotype
    _ = try gnome1.genesis(allocator, 1);

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    // configuration
    var node_activators = [_]neat_math.NodeActivationType{neat_math.NodeActivationType.SigmoidSteepenedActivation};
    var node_activators_prob = [_]f64{1.0};
    var options = Options{
        .node_activators = &node_activators,
        .node_activators_prob = &node_activators_prob,
        .pop_size = 1,
    };

    // init population with one organism
    var pop = try Population.raw_init(allocator);
    defer pop.deinit();
    try pop.spawn(allocator, rand, gnome1, &options);

    var res = try gnome1.mutate_add_node(allocator, rand, pop, &options);
    try std.testing.expect(res);

    // two genes were added, expecting innovation + 2 (3+2)
    try std.testing.expect(pop.next_innov_number.loadUnchecked() == 5);
    try std.testing.expect(pop.innovations.items.len == 1);
    try std.testing.expect(gnome1.genes.len == 5);
    try std.testing.expect(gnome1.nodes.len == 5);

    var added_node = gnome1.nodes[4];
    try std.testing.expect(added_node.id == 6);
    try std.testing.expect(added_node.activation_type == neat_math.NodeActivationType.SigmoidSteepenedActivation);
}

test "Genome mutate link weights" {
    var allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var res = try gnome1.mutate_link_weights(rand, 0.5, 1.0, MutatorType.GaussianMutator);
    try std.testing.expect(res);

    for (gnome1.genes, 0..) |gn, i| {
        // check that link weights are different from original ones (1.5, 2.5, 3.5)
        try std.testing.expect(gn.link.cxn_weight != 1.5 + @as(f64, @floatFromInt(i)));
    }
}

test "Genome mutate random trait" {
    var allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    // configuration
    var options = Options{
        .trait_mut_power = 0.3,
        .trait_param_mut_prob = 0.5,
    };

    var res = try gnome1.mutate_random_trait(rand, &options);
    try std.testing.expect(res);

    var mutation_found = false;
    outer: for (gnome1.traits) |tr| {
        for (tr.params, 0..) |p, pi| {
            if (pi == 0 and p != @as(f64, @floatFromInt(tr.id.?)) / 10) {
                mutation_found = true;
                break :outer;
            } else if (pi > 0 and p != 0) {
                mutation_found = true;
                break :outer;
            }
        }
    }
    try std.testing.expect(mutation_found);
}

test "Genome mutate link trait" {
    var allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var res = try gnome1.mutate_link_trait(rand, 10);
    try std.testing.expect(res);

    var mutation_found = false;
    for (gnome1.genes, 0..) |gn, i| {
        if (gn.link.trait.?.id.? != @as(i64, @intCast(i)) + 1) {
            mutation_found = true;
            break;
        }
    }
    try std.testing.expect(mutation_found);
}

test "Genome mutate node trait" {
    var allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    // add traits to nodes
    for (gnome1.nodes, 0..) |nd, i| {
        if (i < 3) {
            nd.trait = gnome1.traits[i];
        }
    }
    var new_trait = try Trait.init(allocator, 8);
    defer new_trait.deinit();
    new_trait.id = 4;
    for (new_trait.params, 0..) |*p, i| {
        if (i == 0) {
            p.* = 0.4;
        } else {
            p.* = 0;
        }
    }
    gnome1.nodes[3].trait = new_trait;

    var res = try gnome1.mutate_node_trait(rand, 2);
    try std.testing.expect(res);

    var mutation_found = false;
    for (gnome1.nodes, 0..) |nd, i| {
        if (nd.trait.?.id.? != @as(i64, @intCast(i + 1))) {
            mutation_found = true;
            break;
        }
    }
    try std.testing.expect(mutation_found);
}

test "Genome mutate toggle enable" {
    var allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    // add extra connection gene from BIAS to OUT
    var new_gene = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome1.traits[2], 5.5, gnome1.nodes[2], gnome1.nodes[3], false), 4, 0, true);
    var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome1.genes);
    try tmp_genes.append(new_gene);
    gnome1.genes = try tmp_genes.toOwnedSlice();

    var res = try gnome1.mutate_toggle_enable(rand, 50);
    try std.testing.expect(res);

    var mut_count: usize = 0;
    for (gnome1.genes) |gn| {
        if (!gn.is_enabled) {
            mut_count += 1;
        }
    }

    // in our genome only one connection gene can be disabled to not break the network (BIAS -> OUT) because
    // we added extra connection gene to link BIAS and OUT
    try std.testing.expect(mut_count == 1);
}

test "Genome mutate gene re-enable" {
    var allocator = std.testing.allocator;
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();

    // add disabled extra connection gene from BIAS to OUT
    var new_gene = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome1.traits[2], 5.5, gnome1.nodes[2], gnome1.nodes[3], false), 4, 0, false);
    var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome1.genes);
    try tmp_genes.append(new_gene);
    gnome1.genes = try tmp_genes.toOwnedSlice();

    // disable one more gene
    gnome1.genes[1].is_enabled = false;

    var res = try gnome1.mutate_gene_reenable();
    try std.testing.expect(res);

    // verify first encountered disabled gene was enabled and second disabled gene remains unchanged
    try std.testing.expect(gnome1.genes[1].is_enabled);
    try std.testing.expect(gnome1.genes[3].is_enabled == false);
}

test "Genome mate multipoint" {
    var allocator = std.testing.allocator;
    // check equal sized gene pools
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try build_test_genome(allocator, 2);
    defer gnome2.deinit();
    var genome_id: i64 = 3;
    var fitness1: f64 = 1.0;
    var fitness2: f64 = 2.3;

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var genome_child = try gnome1.mate_multipoint(allocator, rand, gnome2, genome_id, fitness1, fitness2);
    defer genome_child.deinit();

    try std.testing.expect(genome_child.genes.len == 3);
    try std.testing.expect(genome_child.nodes.len == 4);
    try std.testing.expect(genome_child.traits.len == 3);

    // check unequal sized gene pools
    var new_gene = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome1.traits[2], 5.5, gnome1.nodes[2], gnome1.nodes[3], false), 4, 0, true);
    var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome1.genes);
    try tmp_genes.append(new_gene);
    gnome1.genes = try tmp_genes.toOwnedSlice();
    fitness1 = 15.0;
    fitness2 = 2.3;

    var genome_child2 = try gnome1.mate_multipoint(allocator, rand, gnome2, genome_id, fitness1, fitness2);
    defer genome_child2.deinit();
    try std.testing.expect(genome_child2.genes.len == 3);
    try std.testing.expect(genome_child2.nodes.len == 4);
    try std.testing.expect(genome_child2.traits.len == 3);
}

test "Genome mate multipoint modular" {
    var allocator = std.testing.allocator;
    // check equal sized gene pools
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try build_test_modular_genome(allocator, 2);
    defer gnome2.deinit();
    var genome_id: i64 = 3;
    var fitness1: f64 = 1.0;
    var fitness2: f64 = 2.3;

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var genome_child = try gnome1.mate_multipoint(allocator, rand, gnome2, genome_id, fitness1, fitness2);
    defer genome_child.deinit();
    try std.testing.expect(genome_child.genes.len == 6);
    try std.testing.expect(genome_child.nodes.len == 7);
    try std.testing.expect(genome_child.traits.len == 3);
    try std.testing.expect(genome_child.control_genes.?.len == 1);
}

test "Genome mate multipoint avg" {
    var allocator = std.testing.allocator;
    // check equal sized gene pools
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try build_test_genome(allocator, 2);
    defer gnome2.deinit();

    var prng = std.rand.DefaultPrng.init(2);
    const rand = prng.random();

    var genome_id: i64 = 3;
    var fitness1: f64 = 1.0;
    var fitness2: f64 = 2.3;
    var genome_child = try gnome1.mate_multipoint_avg(allocator, rand, gnome2, genome_id, fitness1, fitness2);
    defer genome_child.deinit();

    try std.testing.expect(genome_child.genes.len == 3);
    try std.testing.expect(genome_child.nodes.len == 4);
    try std.testing.expect(genome_child.traits.len == 3);

    // check unequal sized gene pools
    var gene = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome1.traits[2], 5.5, gnome1.nodes[2], gnome1.nodes[3], false), 4, 0, false);
    var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome1.genes);
    try tmp_genes.append(gene);
    gnome1.genes = try tmp_genes.toOwnedSlice();
    var gene2 = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome1.traits[2], 5.5, gnome1.nodes[1], gnome1.nodes[3], true), 4, 0, false);
    tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome2.genes);
    try tmp_genes.append(gene2);
    gnome2.genes = try tmp_genes.toOwnedSlice();

    fitness1 = 15.0;
    fitness2 = 2.3;
    var genome_child2 = try gnome1.mate_multipoint_avg(allocator, rand, gnome2, genome_id, fitness1, fitness2);
    defer genome_child2.deinit();

    try std.testing.expect(genome_child2.genes.len == 4);
    try std.testing.expect(genome_child2.nodes.len == 4);
    try std.testing.expect(genome_child2.traits.len == 3);
}

test "Genome mate multipoint avg modular" {
    var allocator = std.testing.allocator;
    // check equal sized gene pools
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try build_test_modular_genome(allocator, 2);
    defer gnome2.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var genome_id: i64 = 3;
    var fitness1: f64 = 1.0;
    var fitness2: f64 = 2.3;
    var genome_child = try gnome1.mate_multipoint_avg(allocator, rand, gnome2, genome_id, fitness1, fitness2);
    defer genome_child.deinit();

    try std.testing.expect(genome_child.genes.len == 6);
    try std.testing.expect(genome_child.nodes.len == 7);
    try std.testing.expect(genome_child.traits.len == 3);
    try std.testing.expect(genome_child.control_genes.?.len == 1);
}

test "Genome mate singlepoint" {
    var allocator = std.testing.allocator;
    // check equal sized gene pools
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try build_test_genome(allocator, 2);
    defer gnome2.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var genome_id: i64 = 3;
    var genome_child = try gnome1.mate_singlepoint(allocator, rand, gnome2, genome_id);
    defer genome_child.deinit();

    try std.testing.expect(genome_child.genes.len == 3);
    try std.testing.expect(genome_child.nodes.len == 4);
    try std.testing.expect(genome_child.traits.len == 3);

    // check unequal sized gene pools
    var gene = try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome1.traits[2], 5.5, gnome1.nodes[2], gnome1.nodes[3], false), 4, 0, false);
    var tmp_genes = std.ArrayList(*Gene).fromOwnedSlice(allocator, gnome1.genes);
    try tmp_genes.append(gene);
    gnome1.genes = try tmp_genes.toOwnedSlice();
    var genome_child2 = try gnome1.mate_singlepoint(allocator, rand, gnome2, genome_id);
    defer genome_child2.deinit();

    try std.testing.expect(genome_child2.genes.len == 3);
    try std.testing.expect(genome_child2.nodes.len == 4);
    try std.testing.expect(genome_child2.traits.len == 3);

    // set second Genome genes to first + one more
    tmp_genes = std.ArrayList(*Gene).init(allocator);
    for (gnome1.genes) |gn| {
        try tmp_genes.append(try Gene.init_copy(allocator, gn, gn.link.trait, gene.link.in_node, gene.link.out_node));
    }
    try tmp_genes.append(try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome1.traits[2], 5.5, gnome1.nodes[2], gnome1.nodes[3], false), 4, 0, false));
    // append additional gene
    try tmp_genes.append(try Gene.init_cxn_gene(allocator, try Link.init_with_trait(allocator, gnome2.traits[2], 5.5, gnome2.nodes[1], gnome2.nodes[3], true), 4, 0, false));
    // free gnome2 old genes
    for (gnome2.genes) |gn| {
        gn.deinit();
    }
    gnome2.allocator.free(gnome2.genes);
    gnome2.genes = try tmp_genes.toOwnedSlice();
    var genome_child3 = try gnome1.mate_singlepoint(allocator, rand, gnome2, genome_id);
    defer genome_child3.deinit();

    try std.testing.expect(genome_child3.genes.len == 4);
    try std.testing.expect(genome_child3.nodes.len == 4);
    try std.testing.expect(genome_child3.traits.len == 3);
}

test "Genome mate singlepoint module" {
    var allocator = std.testing.allocator;
    // check equal sized gene pools
    var gnome1 = try build_test_genome(allocator, 1);
    defer gnome1.deinit();
    var gnome2 = try build_test_modular_genome(allocator, 2);
    defer gnome2.deinit();

    var prng = std.rand.DefaultPrng.init(42);
    const rand = prng.random();

    var genome_id: i64 = 3;
    var genome_child = try gnome1.mate_singlepoint(allocator, rand, gnome2, genome_id);
    defer genome_child.deinit();

    try std.testing.expect(genome_child.genes.len == 6);
    try std.testing.expect(genome_child.nodes.len == 7);
    try std.testing.expect(genome_child.traits.len == 3);
    try std.testing.expect(genome_child.control_genes.?.len == 1);
}
