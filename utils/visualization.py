"""
Visualization utilities for results.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import torch


def plot_vae_reconstruction(original, reconstructed, n_samples=10, save_path=None):
    """
    Plot original vs reconstructed images from VAE.
    
    Args:
        original: Original images (batch_size, 1, 28, 28)
        reconstructed: Reconstructed images (batch_size, 1, 28, 28)
        n_samples: Number of samples to show
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
            
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstruction plot to {save_path}")
    
    plt.show()


def plot_latent_space_2d(vae, dataloader, device, save_path=None):
    """
    Plot 2D latent space (only works if latent_dim=2).
    
    Args:
        vae: Trained VAE model
        dataloader: MNIST dataloader
        device: torch device
        save_path: Optional path to save figure
    """
    if vae.latent_dim != 2:
        print(f"Cannot plot 2D latent space for latent_dim={vae.latent_dim}")
        return
    
    vae.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.view(images.size(0), -1).to(device)
            mu, _ = vae.encode(images)
            latents.append(mu.cpu().numpy())
            labels.append(targets.numpy())
    
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    
    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, 
                         cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('MNIST Latent Space')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved latent space plot to {save_path}")
    
    plt.show()


def plot_diffusion_samples(vae, samples, n_samples=10, save_path=None):
    """
    Plot generated samples from diffusion model.
    
    Args:
        vae: VAE for decoding latents to images
        samples: Latent samples (batch_size, latent_dim)
        n_samples: Number of samples to show
        save_path: Optional path to save figure
    """
    vae.eval()
    
    with torch.no_grad():
        # Decode latents to images
        images = vae.decode(samples[:n_samples])
        images = images.view(-1, 1, 28, 28).cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 1.5, 1.5))
    
    for i in range(n_samples):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    
    plt.suptitle('Generated Samples', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved generated samples to {save_path}")
    
    plt.show()


def plot_training_curves(train_losses, val_losses=None, title='Training Curves', 
                         save_path=None):
    """
    Plot training (and validation) loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 5))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_trajectory(trajectory, latent_dims=[0, 1], save_path=None):
    """
    Plot trajectory in 2D projection of latent space.
    
    Args:
        trajectory: Array of latent states (num_steps, latent_dim)
        latent_dims: Which two dimensions to plot
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    # Plot trajectory
    plt.plot(trajectory[:, latent_dims[0]], 
             trajectory[:, latent_dims[1]],
             'b-', alpha=0.6, linewidth=2)
    
    # Mark start and end
    plt.scatter(trajectory[0, latent_dims[0]], 
               trajectory[0, latent_dims[1]],
               c='green', s=100, marker='o', label='Start (noise)', zorder=5)
    plt.scatter(trajectory[-1, latent_dims[0]], 
               trajectory[-1, latent_dims[1]],
               c='red', s=100, marker='*', label='End (image)', zorder=5)
    
    plt.xlabel(f'Latent Dimension {latent_dims[0]}')
    plt.ylabel(f'Latent Dimension {latent_dims[1]}')
    plt.title('Denoising Trajectory in Latent Space')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    
    plt.show()


def plot_sindy_coefficients(coefficients, feature_names, latent_dim_idx=0, 
                            save_path=None):
    """
    Plot SINDy coefficient matrix as heatmap.
    
    Args:
        coefficients: SINDy coefficient matrix (n_features, n_dims)
        feature_names: Names of library features
        latent_dim_idx: Which latent dimension to plot (or 'all' for heatmap)
        save_path: Optional path to save figure
    """
    if latent_dim_idx == 'all':
        # Plot heatmap of all coefficients
        plt.figure(figsize=(12, 8))
        sns.heatmap(coefficients.T, cmap='RdBu_r', center=0,
                   xticklabels=feature_names, 
                   yticklabels=[f'z_{i}' for i in range(coefficients.shape[1])],
                   cbar_kws={'label': 'Coefficient Value'})
        plt.xlabel('Library Functions')
        plt.ylabel('Latent Dimensions')
        plt.title('SINDy Coefficients')
        plt.tight_layout()
    else:
        # Plot bar chart for single dimension
        plt.figure(figsize=(10, 5))
        nonzero_mask = np.abs(coefficients[latent_dim_idx, :]) > 1e-6
        
        plt.bar(range(len(feature_names)), coefficients[latent_dim_idx, :])
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.xlabel('Library Function')
        plt.ylabel('Coefficient')
        plt.title(f'SINDy Coefficients for z_{latent_dim_idx}')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved coefficient plot to {save_path}")
    
    plt.show()