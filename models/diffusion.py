"""
Simple Diffusion Model operating in VAE latent space.
Predicts noise to remove at each timestep.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleDiffusionModel(nn.Module):
    """
    Simple MLP-based diffusion model for low-dimensional latent space.
    
    Input: noisy latent vector + timestep
    Output: predicted noise
    """
    
    def __init__(self, latent_dim=16, hidden_dim=128, time_emb_dim=32):
        """
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden layer dimensions
            time_emb_dim: Dimension of time embedding
        """
        super(SimpleDiffusionModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        
        # Time embedding (sinusoidal)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Main network
        self.input_layer = nn.Linear(latent_dim, hidden_dim)
        
        self.mid_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        self.output_layer = nn.Linear(hidden_dim, latent_dim)
        
    def positional_encoding(self, t, dim):
        """
        Sinusoidal time embedding.
        
        Args:
            t: Timesteps (batch_size,)
            dim: Embedding dimension
            
        Returns:
            Time embeddings (batch_size, dim)
        """
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, t):
        """
        Predict noise given noisy latent and timestep.
        
        Args:
            x: Noisy latent vectors (batch_size, latent_dim)
            t: Timesteps (batch_size,)
            
        Returns:
            Predicted noise (batch_size, latent_dim)
        """
        # Time embedding
        t_emb = self.positional_encoding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Input processing
        h = F.silu(self.input_layer(x))
        
        # Add time information
        h = h + t_emb
        
        # Middle layers
        h = self.mid_layers(h)
        
        # Output
        return self.output_layer(h)


class DiffusionProcess:
    """
    Handles forward and reverse diffusion process.
    
    Based on DDPM (Denoising Diffusion Probabilistic Models).
    """
    
    def __init__(self, num_timesteps=50, beta_start=1e-4, beta_end=0.02, device='cpu'):
        """
        Args:
            num_timesteps: Total number of diffusion steps
            beta_start: Starting noise schedule value
            beta_end: Ending noise schedule value
            device: torch device
        """
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, device=device
        )
        
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For q(x_t | x_0) - forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For q(x_{t-1} | x_t, x_0) - reverse process posterior
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0).
        Add noise to x_0 to get x_t.
        
        Args:
            x_0: Clean latent (batch_size, latent_dim)
            t: Timesteps (batch_size,)
            noise: Optional pre-generated noise
            
        Returns:
            Noisy latent x_t (batch_size, latent_dim)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t):
        """
        Reverse diffusion: single denoising step.
        
        Args:
            model: Noise prediction model
            x_t: Noisy latent at time t (batch_size, latent_dim)
            t: Current timestep (batch_size,)
            
        Returns:
            x_{t-1}: Less noisy latent (batch_size, latent_dim)
        """
        # Predict noise
        pred_noise = model(x_t, t)
        
        # Get coefficients
        betas_t = self.betas[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None]
        
        # Predict x_0 from x_t and noise
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, return_trajectory=False):
        """
        Generate samples from noise.
        
        Args:
            model: Trained diffusion model
            shape: Shape of samples (batch_size, latent_dim)
            return_trajectory: If True, return all intermediate steps
            
        Returns:
            Sampled latents (and trajectory if requested)
        """
        model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=self.device)
        
        trajectory = [x.cpu().numpy()] if return_trajectory else None
        
        # Iteratively denoise
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
            
            if return_trajectory:
                trajectory.append(x.cpu().numpy())
        
        if return_trajectory:
            return x, np.array(trajectory)
        return x


def diffusion_loss(model, diffusion_process, x_0, t):
    """
    Simple MSE loss between predicted and actual noise.
    
    Args:
        model: Diffusion model
        diffusion_process: DiffusionProcess instance
        x_0: Clean latents (batch_size, latent_dim)
        t: Random timesteps (batch_size,)
        
    Returns:
        loss: MSE between predicted and actual noise
    """
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Add noise to get x_t
    x_t = diffusion_process.q_sample(x_0, t, noise)
    
    # Predict noise
    pred_noise = model(x_t, t)
    
    # MSE loss
    loss = F.mse_loss(pred_noise, noise)
    
    return loss