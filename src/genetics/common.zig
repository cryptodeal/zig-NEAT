const std = @import("std");
const Trait = @import("../trait.zig").Trait;
const NNode = @import("../network/nnode.zig").NNode;
const common = @import("common.zig");
const genome = @import("genome.zig");
const pop_epoch = @import("population_epoch.zig");
const species = @import("species.zig");

// exports
/// The connection Gene.
pub const Gene = @import("gene.zig").Gene;
pub const GenomeError = genome.GenomeError;
pub const Genome = genome.Genome;
pub const ModuleMate = genome.ModuleMate;
pub const Innovation = @import("innovation.zig").Innovation;
/// Multiple-Input Multiple-Output (MIMO) control Gene.
pub const MIMOControlGene = @import("mimo_gene.zig").MIMOControlGene;
/// Genomes and Network along with fitness score.
pub const Organism = @import("organism.zig").Organism;
/// Holds the results of parallel (multi-threaded) reproduction.
pub const ReproductionResult = pop_epoch.ReproductionResult;
/// Single-threaded reproduction cycle runner.
pub const SequentialPopulationEpochExecutor = pop_epoch.SequentialPopulationEpochExecutor;
pub const speciesOrgSort = pop_epoch.speciesOrgSort;
/// Multi-threaded reproduction cycle runner.
pub const ParallelPopulationEpochExecutor = pop_epoch.ParallelPopulationEpochExecutor;
/// Data structure used by `ParallelPopulationEpochExecutor` to
/// store results of reproduction.
pub const WorkerCtx = pop_epoch.WorkerCtx;
/// Population of Organisms and the Species they belong to.
pub const Population = @import("population.zig").Population;
pub const MaxAvgFitness = species.MaxAvgFitness;
/// Data structure holding expected offspring count as well as
/// skim value for Species.
pub const OffspringCount = species.OffspringCount;
/// A group of similar Organisms.
pub const Species = species.Species;

/// The innovation method type to be applied.
pub const InnovationType = enum(u8) {
    NewNodeInnType,
    NewLinkInnType,
};

/// The mutator type specifying a type of mutation of connection weights between `NNode`s.
pub const MutatorType = enum(u8) {
    GaussianMutator,
    GoldGaussianMutator,
};

/// Utility to select Trait with given Id from provided slice of Traits.
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

/// Utility to select NNode with given Id from provided slice NNodes.
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
