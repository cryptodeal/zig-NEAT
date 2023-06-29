const std = @import("std");

const ActivationFn = fn (f64, []f64) f64;
const ModuleActivationFn = fn ([]f64, []f64) []f64;

// neuron activation function types
pub const NodeActivationType = enum {
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
        "SigmoidSteepenedActivation",
        "SigmoidBipolarActivation",
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

    pub fn activate_by_type(input: f64, aux_params: ?[]f64, activation_type: NodeActivationType) !f64 {
        var func: ActivationFn = undefined;
        switch (activation_type) {
            NodeActivationType.SigmoidPlainActivation => func = plain_sigmoid,
            NodeActivationType.SigmoidReducedActivation => func = reduced_sigmoid,
            NodeActivationType.SigmoidBipolarActivation => func = bipolar_sigmoid,
            NodeActivationType.SigmoidSteepenedActivation => func = steepened_sigmoid,
            NodeActivationType.SigmoidApproximationActivation => func = approximation_sigmoid,
            NodeActivationType.SigmoidSteepenedApproximationActivation => func = approximation_steepened_sigmoid,
            NodeActivationType.SigmoidInverseAbsoluteActivation => func = inverse_absolute_sigmoid,
            NodeActivationType.SigmoidLeftShiftedActivation => func = left_shifted_sigmoid,
            NodeActivationType.SigmoidLeftShiftedSteepenedActivation => func = left_shifted_steepened_sigmoid,
            NodeActivationType.SigmoidRightShiftedSteepenedActivation => func = right_shifted_steepened_sigmoid,
            NodeActivationType.TanhActivation => func = hyperbolic_tangent,
            NodeActivationType.GaussianBipolarActivation => func = bipolar_gaussian,
            NodeActivationType.LinearActivation => func = linear,
            NodeActivationType.LinearAbsActivation => func = absolute_linear,
            NodeActivationType.LinearClippedActivation => func = clipped_linear,
            NodeActivationType.NullActivation => func = null_functor,
            NodeActivationType.SignActivation => func = sign_function,
            NodeActivationType.SineActivation => func = sine_function,
            NodeActivationType.StepActivation => func = step_function,
            else => @compileError("unknown activation type: " ++ activation_type),
        }
        return func(input, aux_params);
    }

    pub fn activate_module_by_type(inputs: []f64, aux_params: ?[]f64, activation_type: NodeActivationType) ![]f64 {
        var func: ModuleActivationFn = undefined;
        switch (activation_type) {
            NodeActivationType.MultiplyModuleActivation => func = multiply_module,
            NodeActivationType.MaxModuleActivation => func = max_module,
            NodeActivationType.MinModuleActivation => func = min_module,
            else => @compileError("unknown activation type: " ++ activation_type),
        }
        return func(inputs, aux_params);
    }

    pub fn activation_name_by_type(self: NodeActivationType) []const u8 {
        return ActivationNameTable[@intFromEnum(self)];
    }

    pub fn activation_type_by_name(name: []const u8) NodeActivationType {
        inline for (ActivationNameTable) |enum_name| {
            if (comptime std.mem.eql(u8, name, enum_name)) {
                return @as(NodeActivationType, @enumFromInt(@intFromEnum(enum_name)));
            }
        }
        @compileError(@compileError("unknown activation name: " ++ name));
    }
};

/// plain sigmoid
fn plain_sigmoid(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-input));
}

/// plain reduced sigmoid
fn reduced_sigmoid(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-0.5 * input));
}

/// steepened sigmoid
fn steepened_sigmoid(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-4.924273 * input));
}

/// bipolar sigmoid
fn bipolar_sigmoid(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return (2.0 / (1.0 + @exp(-4.924273 * input))) - 1.0;
}

/// approximation sigmoid w squashing range [-4.0; 4.0]
fn approximation_sigmoid(input: f64, aux_params: []f64) f64 {
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
fn approximation_steepened_sigmoid(input: f64, aux_params: []f64) f64 {
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
fn inverse_absolute_sigmoid(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return 0.5 + (input / (1.0 + @fabs(input))) * 0.5;
}

/// left/right shifted sigmoid
fn left_shifted_sigmoid(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-input - 2.4621365));
}

fn left_shifted_steepened_sigmoid(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-(4.924273 * input + 2.4621365)));
}

fn right_shifted_steepened_sigmoid(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return 1.0 / (1.0 + @exp(-(4.924273 * input - 2.4621365)));
}

/// hyperbolic tangent
fn hyperbolic_tangent(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return std.math.tanh(0.9 * input);
}

/// bipolar Gaussian w xrange->[-1,1] yrange->[-1,1]
fn bipolar_gaussian(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return 2.0 * @exp(-std.math.pow(f64, input * 2.5, 2.0)) - 1.0;
}

/// linear activation
fn linear(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return input;
}

/// absolute linear activation
fn absolute_linear(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return @fabs(input);
}

/// linear activation w clipping; output is linear between [-1; 1] or clipped at -1/1 if outside range
fn clipped_linear(input: f64, aux_params: []f64) f64 {
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
fn null_functor(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    _ = input;
    return 0.0;
}

/// sign activator
fn sign_function(input: f64, aux_params: []f64) f64 {
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
fn sine_function(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    return @sin(2.0 * input);
}

/// step function x<0 ? 0.0 : 1.0
fn step_function(input: f64, aux_params: []f64) f64 {
    _ = aux_params;
    if (std.math.signbit(input)) {
        return 0.0;
    } else {
        return 1.0;
    }
}

/// multiplies input values and returns multiplication result
fn multiply_module(inputs: []f64, aux_params: []f64) []f64 {
    _ = aux_params;
    var res: f64 = 1.0;
    for (inputs) |v| {
        res *= v;
    }
    return []f64{res};
}

/// finds maximal value among inputs and return it
fn max_module(inputs: []f64, aux_params: []f64) []f64 {
    _ = aux_params;
    var max = std.math.f64_min;
    for (inputs) |v| {
        max = @max(max, v);
    }
    return []f64{max};
}

/// finds minimal value among inputs and returns it
fn min_module(inputs: []f64, aux_params: []f64) []f64 {
    _ = aux_params;
    var min = std.math.f64_max;
    for (inputs) |v| {
        min = @min(min, v);
    }
    return []f64{min};
}
