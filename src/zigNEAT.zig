const options = @import("opts.zig");
const trait_common = @import("trait.zig");

/// Experiment data structures and functions, which
/// are used to evaluate NEAT against a given problem.
pub const experiment = @import("experiment/common.zig");

/// Graph data structure and utility functions.
pub const graph = @import("graph/graph.zig");

/// Compositional Pattern Producing Network (CPPN) implementation,
/// which is utilized for Hypercube-based NEAT algorithms.
pub const cppn = @import("cppn/cppn.zig");

/// Genetic Evolution data structures and utility functions.
pub const genetics = @import("genetics/common.zig");

/// Activation functions and math utility functions.
pub const math = @import("math/math.zig");

/// Artificial Neural Network data structures and functions.
pub const network = @import("network/common.zig");

/// Novelty Search data structures and functions.
pub const ns = @import("ns/common.zig");

/// File IO utility functions.
pub const utils = @import("utils/utils.zig");

/// Defines the type of Epoch Executor to use.
pub const EpochExecutorType = options.EpochExecutorType;
/// Defines the method used to calculate Genome compatability.
pub const GenomeCompatibilityMethod = options.GenomeCompatibilityMethod;
/// The NEAT algorithm options.
pub const Options = options.Options;
/// The HyperNEAT algorithm execution options.
pub const HyperNEATContext = options.HyperNEATContext;
/// The ES-HyperNEAT algorithm execution options.
pub const ESHyperNEATContext = options.ESHyperNEATContext;
/// Wraps `std.log` to provide granulated control of logging outputs.
pub const NeatLogger = @import("log.zig");
/// Grouped parameters shared by varying structures during genetic evolution.
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
    _ = cppn;
}
