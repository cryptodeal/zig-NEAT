const std = @import("std");
const Trait = @import("../trait.zig").Trait;
const NNode = @import("../network/nnode.zig").NNode;
const common = @import("common.zig");
const genome = @import("genome.zig");
const pop_epoch = @import("population_epoch.zig");
const species = @import("species.zig");

// exports
pub const Gene = @import("gene.zig").Gene;
pub const GenomeError = genome.GenomeError;
pub const Genome = genome.Genome;
pub const ModuleMate = genome.ModuleMate;
pub const Innovation = @import("innovation.zig").Innovation;
pub const MIMOControlGene = @import("mimo_gene.zig").MIMOControlGene;
pub const Organism = @import("organism.zig").Organism;
pub const ReproductionResult = pop_epoch.ReproductionResult;
pub const SequentialPopulationEpochExecutor = pop_epoch.SequentialPopulationEpochExecutor;
pub const speciesOrgSort = pop_epoch.speciesOrgSort;
pub const ParallelPopulationEpochExecutor = pop_epoch.ParallelPopulationEpochExecutor;
pub const WorkerCtx = pop_epoch.WorkerCtx;
pub const Population = @import("population.zig").Population;
pub const MaxAvgFitness = species.MaxAvgFitness;
pub const OffspringCount = species.OffspringCount;
pub const Species = species.Species;

pub const InnovationType = enum(u8) {
    NewNodeInnType,
    NewLinkInnType,
};

pub const MutatorType = enum(u8) {
    GaussianMutator,
    GoldGaussianMutator,
};

pub const GenomeEncoding = enum(u8) {
    PlainGenomeEncoding,
    YAMLGenomeEncoding,
};

pub fn traitWithId(trait_id: i64, traits: ?[]*Trait) ?*Trait {
    if (trait_id != 0 and traits != null) {
        for (traits.?) |tr| {
            if (tr.id.? == trait_id) {
                return tr;
            }
        }
    }
    return null;
}

pub fn nodeWithId(node_id: i64, nodes: ?[]*NNode) ?*NNode {
    if (node_id != 0 and nodes != null) {
        for (nodes.?) |n| {
            if (n.id == node_id) {
                return n;
            }
        }
    }
    return null;
}

test {
    std.testing.refAllDecls(@This());
}
