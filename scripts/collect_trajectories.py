"""
Step 3: Collect denoising trajectories from diffusion model.

These trajectories will be analyzed with SINDy to discover governing equations.

Usage:
    python scripts/3_collect_trajectories.py
"""
import sys

import torch
import numpy as np
from tqdm import tqdm

from config import (VAE_CONFIG, DIFFUSION_CONFIG, TRAJECTORY_CONFIG, 
                   DEVICE, CHECKPOINT_DIR, RESULTS_DIR, set_seed)
from models import VAE, SimpleDiffusionModel, DiffusionProcess
from utils import plot_trajectory


def collect_single_trajectory(model, diffusion_process, initial_noise=None):
    """
    Collect full denoising trajectory.
    
    Args:
        model: Trained diffusion model
        diffusion_process: DiffusionProcess instance
        initial_noise: Optional initial noise (default: random)
        
    Returns:
        trajectory: Array (num_steps+1, latent_dim) of latent states
    """
    model.eval()
    
    # Start from pure noise
    if initial_noise is None:
        x = torch.randn(1, DIFFUSION_CONFIG['latent_dim'], device=DEVICE)
    else:
        x = initial_noise
    
    trajectory = [x.cpu().numpy().squeeze()]
    
    # Iteratively denoise
    with torch.no_grad():
        for i in reversed(range(diffusion_process.num_timesteps)):
            t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
            x = diffusion_process.p_sample(model, x, t)
            trajectory.append(x.cpu().numpy().squeeze())
    
    return np.array(trajectory)


def main():
    """Main trajectory collection."""
    set_seed()
    
    print("="*80)
    print("STEP 3: Collecting Denoising Trajectories")
    print("="*80)
    print(f"Number of trajectories: {TRAJECTORY_CONFIG['num_trajectories']}")
    print(f"Trajectories per class: {TRAJECTORY_CONFIG['trajectories_per_class']}")
    print(f"Timesteps per trajectory: {DIFFUSION_CONFIG['num_timesteps'] + 1}")
    print("="*80)
    
    # Load trained models
    print("\nLoading trained models...")
    
    # Load VAE (for visualization later)
    vae = VAE(
        input_dim=VAE_CONFIG['input_dim'],
        hidden_dim=VAE_CONFIG['hidden_dim'],
        latent_dim=VAE_CONFIG['latent_dim']
    ).to(DEVICE)
    vae_checkpoint = torch.load(CHECKPOINT_DIR / "vae_best.pt", map_location=DEVICE)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    print("✓ VAE loaded")
    
    # Load diffusion model
    diffusion_model = SimpleDiffusionModel(
        latent_dim=DIFFUSION_CONFIG['latent_dim'],
        hidden_dim=DIFFUSION_CONFIG['hidden_dim']
    ).to(DEVICE)
    diff_checkpoint = torch.load(CHECKPOINT_DIR / "diffusion_best.pt", map_location=DEVICE)
    diffusion_model.load_state_dict(diff_checkpoint['model_state_dict'])
    diffusion_model.eval()
    print("✓ Diffusion model loaded")
    
    # Create diffusion process
    diffusion_process = DiffusionProcess(
        num_timesteps=DIFFUSION_CONFIG['num_timesteps'],
        beta_start=DIFFUSION_CONFIG['beta_start'],
        beta_end=DIFFUSION_CONFIG['beta_end'],
        device=DEVICE
    )
    print("✓ Diffusion process created")
    
    # Collect trajectories
    print("\nCollecting trajectories...")
    all_trajectories = []
    
    for i in tqdm(range(TRAJECTORY_CONFIG['num_trajectories']), desc='Trajectories'):
        trajectory = collect_single_trajectory(diffusion_model, diffusion_process)
        all_trajectories.append(trajectory)
    
    all_trajectories = np.array(all_trajectories)
    print(f"\nCollected trajectories shape: {all_trajectories.shape}")
    print(f"  (num_trajectories, num_steps, latent_dim)")
    
    # Save trajectories
    save_path = TRAJECTORY_CONFIG['save_path']
    np.savez(
        save_path,
        trajectories=all_trajectories,
        config={
            'num_trajectories': TRAJECTORY_CONFIG['num_trajectories'],
            'num_timesteps': DIFFUSION_CONFIG['num_timesteps'],
            'latent_dim': DIFFUSION_CONFIG['latent_dim'],
            'beta_start': DIFFUSION_CONFIG['beta_start'],
            'beta_end': DIFFUSION_CONFIG['beta_end'],
        }
    )
    print(f"\n✓ Saved trajectories to: {save_path}")
    
    # Compute statistics
    print("\nTrajectory Statistics:")
    print(f"  Mean latent norm (start): {np.linalg.norm(all_trajectories[:, 0, :], axis=1).mean():.3f}")
    print(f"  Mean latent norm (end):   {np.linalg.norm(all_trajectories[:, -1, :], axis=1).mean():.3f}")
    print(f"  Std latent norm (start):  {np.linalg.norm(all_trajectories[:, 0, :], axis=1).std():.3f}")
    print(f"  Std latent norm (end):    {np.linalg.norm(all_trajectories[:, -1, :], axis=1).std():.3f}")
    
    # Visualize a few trajectories
    print("\nVisualizing sample trajectories...")
    
    for i in range(min(3, len(all_trajectories))):
        plot_trajectory(
            all_trajectories[i],
            latent_dims=[0, 1],
            save_path=RESULTS_DIR / f'trajectory_{i}.png'
        )
    
    # Show variation across trajectories
    print("\nVariation across trajectories at different timesteps:")
    for step in [0, DIFFUSION_CONFIG['num_timesteps']//2, DIFFUSION_CONFIG['num_timesteps']]:
        latents_at_step = all_trajectories[:, step, :]
        mean_latent = latents_at_step.mean(axis=0)
        std_latent = latents_at_step.std(axis=0)
        print(f"  Step {step:2d} - Mean: {mean_latent.mean():.3f}, Std: {std_latent.mean():.3f}")
    
    print("\n" + "="*80)
    print("Trajectory collection complete!")
    print(f"Saved {len(all_trajectories)} trajectories")
    print("="*80)
    
    print("\nTrajectory collection completed successfully!")


if __name__ == '__main__':
    main()