// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Differentiable element functions of the training layers. The backward
// shaders get their derivatives from Slang's automatic differentiation
// (bwd_diff), so a forward function and its gradient cannot drift apart.
module differentiable;

// An element-wise activation, the same function as the forward layer's shader.
public interface IActivation {
    [Differentiable]
    static float apply(float x);
}

// layers::relu: max(x, 0).
public struct ReluActivation : IActivation {
    [Differentiable]
    public static float apply(float x) { return max(x, 0.0); }
}

// layers::unary(UnaryOp::Sigmoid): 1 / (1 + exp(-x)).
public struct SigmoidActivation : IActivation {
    [Differentiable]
    public static float apply(float x) { return 1.0 / (1.0 + exp(-x)); }
}

// The squared error of a sigmoid output against its target.
[Differentiable]
public float sigmoidSquaredError(float logit, no_diff float target) {
    const float error = SigmoidActivation.apply(logit) - target;
    return error * error;
}

// The gradient of `Activation` at `x` times the output gradient `outputGradient`.
public float activationGradient<Activation : IActivation>(float x, float outputGradient) {
    var input = diffPair(x);
    bwd_diff(Activation.apply)(input, outputGradient);
    return input.d;
}
