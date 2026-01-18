"""
Step 1: Train VAE to compress MNIST into low-dimensional latent space.

Usage:
    python scripts/1_train_vae.py
"""
import sys

import torch
from tqdm import tqdm
import numpy as np

from config import VAE_CONFIG, DEVICE, CHECKPOINT_DIR, RESULTS_DIR, set_seed
from models import VAE, vae_loss
from utils import get_mnist_dataloaders, flatten_image, unflatten_image
from utils import plot_vae_reconstruction, plot_training_curves


def train_epoch(model, dataloader, optimizer, device, beta=1.0):
    """Train VAE for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for images, _ in tqdm(dataloader, desc='Training'):
        # Flatten images
        images = flatten_image(images).to(device)
        
        # Forward pass
        recon, mu, logvar = model(images)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    n_samples = len(dataloader.dataset)
    return (total_loss / n_samples, 
            total_recon / n_samples, 
            total_kl / n_samples)


@torch.no_grad()
def evaluate(model, dataloader, device, beta=1.0):
    """Evaluate VAE on test set."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for images, _ in dataloader:
        images = flatten_image(images).to(device)
        recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta)
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    n_samples = len(dataloader.dataset)
    return (total_loss / n_samples,
            total_recon / n_samples,
            total_kl / n_samples)


def main():
    """Main training loop."""
    # Set seed for reproducibility
    set_seed()
    
    print("="*80)
    print("STEP 1: Training VAE")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Latent dimension: {VAE_CONFIG['latent_dim']}")
    print(f"Hidden dimension: {VAE_CONFIG['hidden_dim']}")
    print(f"Epochs: {VAE_CONFIG['epochs']}")
    print("="*80)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=VAE_CONFIG['batch_size']
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating VAE model...")
    model = VAE(
        input_dim=VAE_CONFIG['input_dim'],
        hidden_dim=VAE_CONFIG['hidden_dim'],
        latent_dim=VAE_CONFIG['latent_dim']
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=VAE_CONFIG['learning_rate']
    )
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    for epoch in range(1, VAE_CONFIG['epochs'] + 1):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, DEVICE, VAE_CONFIG['beta']
        )
        
        # Evaluate
        test_loss, test_recon, test_kl = evaluate(
            model, test_loader, DEVICE, VAE_CONFIG['beta']
        )
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Print progress
        print(f"\nEpoch {epoch}/{VAE_CONFIG['epochs']}")
        print(f"  Train - Total: {train_loss:.2f}, Recon: {train_recon:.2f}, KL: {train_kl:.2f}")
        print(f"  Test  - Total: {test_loss:.2f}, Recon: {test_recon:.2f}, KL: {test_kl:.2f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint_path = CHECKPOINT_DIR / "vae_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'config': VAE_CONFIG
            }, checkpoint_path)
            print(f"  ✓ Saved best model (test loss: {test_loss:.2f})")
    
    # Save final model
    final_checkpoint_path = CHECKPOINT_DIR / "vae_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'config': VAE_CONFIG
    }, final_checkpoint_path)
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best test loss: {best_test_loss:.2f}")
    print(f"Saved checkpoints to: {CHECKPOINT_DIR}")
    print("="*80)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # 1. Training curves
    plot_training_curves(
        train_losses, test_losses,
        title='VAE Training Curves',
        save_path=RESULTS_DIR / 'vae_training_curves.png'
    )
    
    # 2. Reconstructions
    model.eval()
    test_images, _ = next(iter(test_loader))
    test_images_flat = flatten_image(test_images).to(DEVICE)
    
    with torch.no_grad():
        recon, _, _ = model(test_images_flat)
    
    recon_images = unflatten_image(recon).cpu()
    
    plot_vae_reconstruction(
        test_images, recon_images,
        n_samples=10,
        save_path=RESULTS_DIR / 'vae_reconstructions.png'
    )
    
    print("\nVAE training completed successfully!")


if __name__ == '__main__':
    main()