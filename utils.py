import torch
import numpy as np

def generate_collocation_points(n_points):
    x = torch.rand(n_points, 1)
    t = torch.rand(n_points, 1)
    return torch.cat([x, t], dim=1)

def generate_boundary_points(n_points):
    t = torch.rand(n_points, 1)
    xb = torch.zeros_like(t)
    return torch.cat([xb, t], dim=1)

def generate_initial_points(n_points):
    x = torch.rand(n_points, 1)
    t0 = torch.zeros_like(x)
    return torch.cat([x, t0], dim=1)