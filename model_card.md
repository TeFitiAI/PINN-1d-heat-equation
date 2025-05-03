---
tags:
- pinn
- physics-informed
- 1d heat equation
- gradio
- scientific machine learning
---

# PINN for 1D Heat Equation

This is a Physics-Informed Neural Network (PINN) trained to solve the 1D heat equation:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

## How it works
- Uses PyTorch to define and train a neural net to approximate the solution.
- Trained on collocation points in domain with physics loss, boundary, and initial conditions.
- Deployed with Gradio to predict `u(x,t)`.

## Try it live
Use the sliders to input values of `x` and `t` and get the temperature prediction!