const std = @import("std");

const ActivationFn = fn (f64, []f64) f64;
const ModuleActivationFn = fn ([]f64, []f64) []f64;

// neuron activation function types
pub const NodeActivationType = enum(usize) {
    // sigmoid activation functions
    SigmoidPlainActivation,
    SigmoidReducedActivation,
    SigmoidBipolarActivation,
    SigmoidSteepenedActivation,
    SigmoidApproximationActivation,
    SigmoidSteepenedApproximationActivation,
    SigmoidInverseAbsoluteActivation,
    SigmoidLeftShiftedActivation,
    SigmoidLeftShiftedSteepenedActivation,
    SigmoidRightShiftedSteepenedActivation,

    // other activation functions
    TanhActivation,
    GaussianBipolarActivation,
    LinearActivation,
    LinearAbsActivation,
    LinearClippedActivation,
    NullActivation,
    SignActivation,
    SineActivation,
    StepActivation,

    // modular activation functions
    MultiplyModuleActivation,
    MaxModuleActivation,
    MinModuleActivation,

    pub const ActivationNameTable = [@typeInfo(NodeActivationType).Enum.fields.len][]const u8{
        // sigmoid activation functions
        "SigmoidPlainActivation",
        "SigmoidReducedActivation",
        "SigmoidBipolarActivation",
        "SigmoidSteepenedActivation",
        "SigmoidApproximationActivation",
        "SigmoidSteepenedApproximationActivation",
        "SigmoidInverseAbsoluteActivation",
        "SigmoidLeftShiftedActivation",
        "SigmoidLeftShiftedSteepenedActivation",
        "SigmoidRightShiftedSteepenedActivation",
        // other activation functions
        "TanhActivation",
        "GaussianBipolarActivation",
        "LinearActivation",
        "LinearAbsActivation",
        "LinearClippedActivation",
        "NullActivation",
        "SignActivation",
        "SineActivation",
        "StepActivation",
        // modular activation functions
        "MultiplyModuleActivation",
        "MaxModuleActivation",
        "MinModuleActivation",
    };

    pub fn activateByType(input: f64, aux_params: ?[]f64, activation_type: NodeActivationType) !f64 {
        return switch (activation_type) {
            NodeActivationType.SigmoidPlainActivation => plainSigmoid(input, aux_params),
            NodeActivationType.SigmoidReducedActivation => reducedSigmoid(input, aux_params),
            NodeActivationType.SigmoidBipolarActivation => bipolarSigmoid(input, aux_params),
            NodeActivationType.SigmoidSteepenedActivation => steepenedSigmoid(input, aux_params),
            NodeActivationType.SigmoidApproximationActivation => approximationSigmoid(input, aux_params),
            NodeActivationType.SigmoidSteepenedApproximationActivation => approximationSteepenedSigmoid(input, aux_params),
            NodeActivationType.SigmoidInverseAbsoluteActivation => inverseAbsoluteSigmoid(input, aux_params),
            NodeActivationType.SigmoidLeftShiftedActivation => leftShiftedSigmoid(input, aux_params),
            NodeActivationType.SigmoidLeftShiftedSteepenedActivation => leftShiftedSteepenedSigmoid(input, aux_params),
            NodeActivationType.SigmoidRightShiftedSteepenedActivation => rightShiftedSteepenedSigmoid(input, aux_params),
            NodeActivationType.TanhActivation => hyperbolicTangent(input, aux_params),
            NodeActivationType.GaussianBipolarActivation => bipolarGaussian(input, aux_params),
            NodeActivationType.LinearActivation => linear(input, aux_params),
            NodeActivationType.LinearAbsActivation => absoluteLinear(input, aux_params),
            NodeActivationType.LinearClippedActivation => clippedLinear(input, aux_params),
            NodeActivationType.NullActivation => nullFunctor(input, aux_params),
            NodeActivationType.SignActivation => signFunction(input, aux_params),
            NodeActivationType.SineActivation => sineFunction(input, aux_params),
            NodeActivationType.StepActivation => stepFunction(input, aux_params),
            else => error.UnknownNodeActivationType,
        };
    }

    pub fn activateModuleByType(inputs: []f64, aux_params: ?[]f64, activation_type: NodeActivationType) ![1]f64 {
        return switch (activation_type) {
            NodeActivationType.MultiplyModuleActivation => multiplyModule(inputs, aux_params),
            NodeActivationType.MaxModuleActivation => maxModule(inputs, aux_params),
            NodeActivationType.MinModuleActivation => minModule(inputs, aux_params),
            else => error.UnknownModuleActivationType,
        };
    }

    pub fn activationNameByType(self: NodeActivationType) []const u8 {
        return @tagName(self);
    }

    pub fn activationTypeByName(name: []const u8) NodeActivationType {
        inline for (ActivationNameTable, 0..) |enum_name, idx| {
            if (std.mem.eql(u8, name, enum_name)) {
                return @as(NodeActivationType, @enumFromInt(idx));
            }
        }
        unreachable;
    }
};

/// plain sigmoid
fn plainSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-input));
}

/// plain reduced sigmoid
fn reducedSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-0.5 * input));
}

/// steepened sigmoid
fn steepenedSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-4.924273 * input));
}

/// bipolar sigmoid
fn bipolarSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return (2.0 / (1.0 + @exp(-4.924273 * input))) - 1.0;
}

/// approximation sigmoid w squashing range [-4.0; 4.0]
fn approximationSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    if (input < -4.0) {
        return 0.0;
    } else if (input < 0.0) {
        return (input + 4.0) * (input + 4.0) * (1 / 32);
    } else if (input < 4.0) {
        return 1.0 - (input - 4.0) * (input - 4.0) * (1 / 32);
    } else {
        return 1.0;
    }
}

/// approximation steepened sigmoid w squashing range [-1.0; 1.0]
fn approximationSteepenedSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    if (input < -1.0) {
        return 0.0;
    } else if (input < 0.0) {
        return (input + 1.0) * (input + 1.0) * 0.5;
    } else if (input < 1.0) {
        return 1.0 - (input - 1.0) * (input - 1.0) * 0.5;
    } else {
        return 1.0;
    }
}

/// inverse absolute sigmoid
fn inverseAbsoluteSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return 0.5 + (input / (1.0 + @fabs(input))) * 0.5;
}

/// left/right shifted sigmoid
fn leftShiftedSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-input - 2.4621365));
}

fn leftShiftedSteepenedSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-(4.924273 * input + 2.4621365)));
}

fn rightShiftedSteepenedSigmoid(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-(4.924273 * input - 2.4621365)));
}

/// hyperbolic tangent
fn hyperbolicTangent(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return std.math.tanh(0.9 * input);
}

/// bipolar Gaussian w xrange->[-1,1] yrange->[-1,1]
fn bipolarGaussian(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return 2.0 * @exp(-std.math.pow(f64, input * 2.5, 2.0)) - 1.0;
}

/// linear activation
fn linear(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return input;
}

/// absolute linear activation
fn absoluteLinear(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return @fabs(input);
}

/// linear activation w clipping; output is linear between [-1; 1] or clipped at -1/1 if outside range
fn clippedLinear(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    if (input < -1.0) {
        return -1.0;
    }
    if (input > 1.0) {
        return 1.0;
    }
    return input;
}

/// null activator
fn nullFunctor(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    _ = input;
    return 0.0;
}

/// sign activator
fn signFunction(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    if (std.math.isNan(input) or input == 0.0) {
        return 0.0;
    } else if (std.math.signbit(input)) {
        return -1.0;
    } else {
        return 1.0;
    }
}

/// sine periodic activation with doubled period
fn sineFunction(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    return @sin(2.0 * input);
}

/// step function x < 0 ? 0.0 : 1.0
fn stepFunction(input: f64, aux_params: ?[]f64) f64 {
    _ = aux_params;
    if (std.math.signbit(input)) {
        return 0.0;
    } else {
        return 1.0;
    }
}

/// multiplies input values and returns multiplication result
fn multiplyModule(inputs: []f64, aux_params: ?[]f64) [1]f64 {
    _ = aux_params;
    var res: f64 = 1.0;
    for (inputs) |v| {
        res *= v;
    }
    var out = [1]f64{res};
    return out;
}

/// finds maximal value among inputs and return it
fn maxModule(inputs: []f64, aux_params: ?[]f64) [1]f64 {
    _ = aux_params;
    var max = std.math.floatMin(f64);
    for (inputs) |v| {
        max = @max(max, v);
    }
    var out = [1]f64{max};
    return out;
}

/// finds minimal value among inputs and returns it
fn minModule(inputs: []f64, aux_params: ?[]f64) [1]f64 {
    _ = aux_params;
    var min = std.math.floatMax(f64);
    for (inputs) |v| {
        min = @min(min, v);
    }
    var out = [1]f64{min};
    return out;
}
