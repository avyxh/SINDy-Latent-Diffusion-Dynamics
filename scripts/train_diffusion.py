"""
Step 2: Train diffusion model in VAE latent space.

Usage:
    python scripts/2_train_diffusion.py
"""
import sys

import torch
from tqdm import tqdm

from config import (VAE_CONFIG, DIFFUSION_CONFIG, DEVICE, CHECKPOINT_DIR, 
                   RESULTS_DIR, set_seed, DATA_DIR)
from models import VAE, SimpleDiffusionModel, DiffusionProcess, diffusion_loss
from utils import get_mnist_dataloaders, flatten_image
from utils import plot_training_curves, plot_diffusion_samples


def get_latent_dataset(vae, dataloader, device):
    """
    Encode entire dataset into latent space.
    
    Returns:
        latents: Tensor of latent codes (n_samples, latent_dim)
    """
    vae.eval()
    latents_list = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Encoding to latent space'):
            images = flatten_image(images).to(device)
            mu, _ = vae.encode(images)  # Use mean, not sampled
            latents_list.append(mu.cpu())
    
    return torch.cat(latents_list, dim=0)


def train_epoch(model, diffusion_process, latent_loader, optimizer, device):
    """Train diffusion model for one epoch."""
    model.train()
    total_loss = 0
    
    for latents in tqdm(latent_loader, desc='Training'):
        latents = latents.to(device)
        batch_size = latents.size(0)
        
        # Random timesteps
        t = torch.randint(0, diffusion_process.num_timesteps, (batch_size,), device=device)
        
        # Compute loss
        loss = diffusion_loss(model, diffusion_process, latents, t)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_size
    
    return total_loss / len(latent_loader.dataset)


def main():
    """Main training loop."""
    set_seed()
    
    print("="*80)
    print("STEP 2: Training Diffusion Model in Latent Space")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Latent dimension: {DIFFUSION_CONFIG['latent_dim']}")
    print(f"Diffusion steps: {DIFFUSION_CONFIG['num_timesteps']}")
    print(f"Epochs: {DIFFUSION_CONFIG['epochs']}")
    print("="*80)
    
    # Load trained VAE
    print("\nLoading trained VAE...")
    vae = VAE(
        input_dim=VAE_CONFIG['input_dim'],
        hidden_dim=VAE_CONFIG['hidden_dim'],
        latent_dim=VAE_CONFIG['latent_dim']
    ).to(DEVICE)
    
    vae_checkpoint = torch.load(CHECKPOINT_DIR / "vae_best.pt", map_location=DEVICE)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    print("✓ VAE loaded successfully")
    
    # Load MNIST and encode to latent space
    print("\nEncoding MNIST to latent space...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=VAE_CONFIG['batch_size'],
        data_dir=DATA_DIR
    )
    
    train_latents = get_latent_dataset(vae, train_loader, DEVICE)
    test_latents = get_latent_dataset(vae, test_loader, DEVICE)
    
    print(f"Training latents shape: {train_latents.shape}")
    print(f"Test latents shape: {test_latents.shape}")
    
    # Create latent dataloaders
    train_latent_loader = torch.utils.data.DataLoader(
        train_latents,
        batch_size=DIFFUSION_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0  # Already in memory
    )
    
    test_latent_loader = torch.utils.data.DataLoader(
        test_latents,
        batch_size=DIFFUSION_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create diffusion model
    print("\nCreating diffusion model...")
    model = SimpleDiffusionModel(
        latent_dim=DIFFUSION_CONFIG['latent_dim'],
        hidden_dim=DIFFUSION_CONFIG['hidden_dim']
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create diffusion process
    diffusion_process = DiffusionProcess(
        num_timesteps=DIFFUSION_CONFIG['num_timesteps'],
        beta_start=DIFFUSION_CONFIG['beta_start'],
        beta_end=DIFFUSION_CONFIG['beta_end'],
        device=DEVICE
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DIFFUSION_CONFIG['learning_rate']
    )
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    best_train_loss = float('inf')
    
    for epoch in range(1, DIFFUSION_CONFIG['epochs'] + 1):
        # Train
        train_loss = train_epoch(
            model, diffusion_process, train_latent_loader, optimizer, DEVICE
        )
        
        train_losses.append(train_loss)
        
        print(f"\nEpoch {epoch}/{DIFFUSION_CONFIG['epochs']}")
        print(f"  Train Loss: {train_loss:.6f}")
        
        # Save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            checkpoint_path = CHECKPOINT_DIR / "diffusion_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': DIFFUSION_CONFIG
            }, checkpoint_path)
            print(f"  ✓ Saved best model (train loss: {train_loss:.6f})")
    
    # Save final model
    final_checkpoint_path = CHECKPOINT_DIR / "diffusion_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'config': DIFFUSION_CONFIG
    }, final_checkpoint_path)
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best train loss: {best_train_loss:.6f}")
    print(f"Saved checkpoints to: {CHECKPOINT_DIR}")
    print("="*80)
    
    # Generate samples
    print("\nGenerating sample images...")
    model.eval()
    
    samples = diffusion_process.sample(
        model,
        shape=(16, DIFFUSION_CONFIG['latent_dim']),
        return_trajectory=False
    )
    
    plot_diffusion_samples(
        vae, samples,
        n_samples=10,
        save_path=RESULTS_DIR / 'diffusion_samples.png'
    )
    
    # Training curves
    plot_training_curves(
        train_losses,
        title='Diffusion Training Curves',
        save_path=RESULTS_DIR / 'diffusion_training_curves.png'
    )
    
    print("\nDiffusion training completed successfully!")


if __name__ == '__main__':
    main()