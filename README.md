# Sparse Equation Discovery for Diffusion Model Interpretability

**Discovering interpretable mathematical equations governing image generation dynamics in diffusion models**

## Overview

Diffusion models have revolutionized generative AI, but remain largely black-box systems with billions of parameters. This project applies **Sparse Identification of Nonlinear Dynamics (SINDy)** to discover interpretable mathematical equations that govern how diffusion models generate images, bridging the gap between machine learning and physical dynamical systems.

### Key Insight

Just as SINDy has been successfully applied to discover PDEs for physical Brownian motion, the same methodology is applied to **ML diffusion models**—revealing that these generative processes follow sparse, interpretable mathematical structures despite their apparent complexity.

---

## Motivation

**Current Problem:**
- Diffusion models use massive neural networks (millions of parameters)
- No interpretable understanding of *how* generation actually works
- Difficult to control, debug, or improve systematically

**Approach:**
- Work in compressed 16D latent space (not raw pixels)
- Collect denoising trajectories showing how latent codes evolve
- Apply sparse regression to discover governing equations
- Extract interpretable mathematical structure from black-box process

---

## Project Pipeline

### Phase 1: Variational Autoencoder (VAE)
- Train VAE to compress MNIST images (28×28 pixels → 16D latent space)
- Creates interpretable, low-dimensional representation
- Enables tractable dynamics discovery

### Phase 2: Latent Diffusion Model
- Train diffusion model operating in 16D latent space
- Forward process: add Gaussian noise to latent codes
- Reverse process: neural network learns to denoise
- 50-step denoising trajectory from noise → image

### Phase 3: Trajectory Collection
- Generate 100+ complete denoising trajectories
- Save latent codes at each timestep: z₀, z₁, ..., z₅₀
- Compute velocities: ż ≈ (z_{t+1} - z_{t-1}) / 2Δt
- Creates dataset of (latent position, velocity, time) tuples

### Phase 4: SINDy Analysis
- Build custom function library with 1259 candidate terms per dimension:
  - Polynomial terms: 1, z_i, z_i², z_i³, z_i·z_j
  - Time interactions: t, t², t·z_i, t²·z_i
  - Exponentials: exp(-t·z_i), exp(-z_i²)
- Apply sparse regression: ż_i = Σ θ_{ik} φ_k(z, t)
- Enforce sparsity: most coefficients forced to zero
- Extract minimal equation set describing dynamics
