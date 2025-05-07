import torch
import torch.nn as nn
import numpy as np   # ✅ Add this line
import matplotlib.pyplot as plt
from model import PINN
from utils import generate_collocation_points, generate_boundary_points, generate_initial_points


def initial_condition(x):
    return torch.sin(np.pi * x)

def train():
    model = PINN(layers=[2, 32, 32, 32, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    alpha = 0.01

    collocation = generate_collocation_points(1000).requires_grad_(True)
    initial = generate_initial_points(100).requires_grad_(True)
    boundary = generate_boundary_points(100).requires_grad_(True)

    x0 = initial[:, 0:1]
    u0 = initial_condition(x0)

    for epoch in range(5000):
        optimizer.zero_grad()

        # PDE Loss
        xt = collocation.clone()
        xt.requires_grad_(True)
        u = model(xt)
        u_t = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0][:, 1:2]
        u_x = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        f = u_t - alpha * u_xx
        loss_pde = torch.mean(f**2)

        # Initial Loss
        u_init = model(initial)
        loss_ic = torch.mean((u_init - u0) ** 2)

        # Boundary Loss
        u_b = model(boundary)
        loss_bc = torch.mean((u_b) ** 2)

        loss = loss_pde + loss_ic + loss_bc
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "outputs/trained_model.pt")

    # Optional: plot training loss
    # (Not recorded here for brevity)

if __name__ == "__main__":
    train()