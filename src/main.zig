const options = @import("opts.zig");
const trait_common = @import("trait.zig");

// export all Experiment Structs/Types/Fns
pub const experiment = @import("experiment/common.zig");

// export all Graph Structs/Types/Fns
pub const graph = @import("graph/graph.zig");

// export all zig-NEAT Genetic Structs/Types/Fns
pub const genetics = @import("genetics/common.zig");

// export all zig-NEAT Math/Activation Structs/Types/Fns
pub const math = @import("math/math.zig");

// export all zig-NEAT Network Structs/Types/Fns
pub const network = @import("network/common.zig");

// export all zig-NEAT Novelty Search Structs/Types/Fns
pub const ns = @import("ns/common.zig");

// export other zig-NEAT Structs/Types/Fns
pub const EpochExecutorType = options.EpochExecutorType;
pub const GenomeCompatibilityMethod = options.GenomeCompatibilityMethod;
pub const Options = options.Options;
pub const NeatLogger = @import("log.zig");
pub const Trait = trait_common.Trait;

test {
    _ = genetics;
    _ = experiment;
    _ = graph;
    _ = math;
    _ = network;
    _ = ns;
    _ = trait_common;
    _ = options;
}
