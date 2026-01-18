"""
Generate new MNIST digits using discovered SINDy equations.

This tests whether the sparse equations capture enough structure
to generate recognizable images.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import solve_ivp
from pathlib import Path

from config import RESULTS_DIR, VAE_CONFIG, DEVICE
from models.vae import VAE

# Must define these for unpickling the model
def exp_decay(x):
    return np.exp(-np.abs(x))

def gaussian(x):
    return np.exp(-x**2)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def name_exp(x): return f"exp(-|{x}|)"
def name_gauss(x): return f"exp(-{x}^2)"
def name_sigmoid(x): return f"sigmoid({x})"


def load_models():
    """Load SINDy model and VAE decoder."""
    print("Loading models...")
    
    # Load SINDy
    with open(RESULTS_DIR / 'sindy_model.pkl', 'rb') as f:
        sindy_model = pickle.load(f)
    
    # Load VAE
    vae = VAE(
        latent_dim=VAE_CONFIG['latent_dim'],
        hidden_dim=VAE_CONFIG['hidden_dim']
    ).to(DEVICE)
    
    checkpoint = torch.load(
        Path('checkpoints') / 'vae_best.pt',
        map_location=DEVICE
    )
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    print("✓ Models loaded!")
    return sindy_model, vae


def generate_with_sindy(sindy_model, vae, n_samples=16):
    """
    Generate digits by integrating SINDy equations.
    
    Args:
        sindy_model: Fitted SINDy model
        vae: Trained VAE for decoding
        n_samples: Number of digits to generate
    
    Returns:
        images: Generated images (n_samples, 28, 28)
        trajectories: Latent trajectories (n_samples, n_steps, 16)
    """
    print(f"\nGenerating {n_samples} digits with SINDy equations...")
    
    latent_dim = VAE_CONFIG['latent_dim']
    
    # Define ODE function
    def sindy_dynamics(t, z):
        """Dynamics from discovered equations."""
        z_augmented = np.concatenate([z, [t]]).reshape(1, -1)
        dzdt = sindy_model.predict(z_augmented)
        return dzdt.flatten()
    
    images = []
    trajectories = []
    
    for i in range(n_samples):
        # 1. Start from noise
        z_noise = np.random.randn(latent_dim) * 4.0  # Scale similar to training
        
        # 2. Integrate SINDy equations from t=0 (noise) to t=1 (image)
        t_span = (0, 1)
        t_eval = np.linspace(0, 1, 51)
        
        sol = solve_ivp(
            sindy_dynamics,
            t_span,
            z_noise,
            t_eval=t_eval,
            method='RK45',
            max_step=0.02
        )
        
        # 3. Final latent code
        z_final = sol.y[:, -1]
        
        # 4. Decode with VAE
        with torch.no_grad():
            z_tensor = torch.tensor(z_final, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            image = vae.decode(z_tensor).cpu().numpy().reshape(28, 28)
        
        images.append(image)
        trajectories.append(sol.y.T)  # (n_steps, latent_dim)
        
        if (i + 1) % 4 == 0:
            print(f"  Generated {i+1}/{n_samples}...")
    
    images = np.array(images)
    # Keep trajectories as list since they may have different lengths
    return images, trajectories


def generate_with_diffusion(vae, diffusion_model, diffusion_process, n_samples=16):
    """
    Generate digits with original diffusion model for comparison.
    
    Args:
        vae: VAE model
        diffusion_model: Trained diffusion model
        diffusion_process: Diffusion process object
        n_samples: Number to generate
    
    Returns:
        images: Generated images
    """
    print(f"\nGenerating {n_samples} digits with neural diffusion...")
    
    from models.diffusion import DiffusionProcess
    from config import DIFFUSION_CONFIG
    
    # Load diffusion model
    diffusion_model_module = __import__('models.diffusion', fromlist=['UNet'])
    UNet = diffusion_model_module.UNet
    
    model = UNet(
        latent_dim=DIFFUSION_CONFIG['latent_dim'],
        hidden_dim=DIFFUSION_CONFIG['hidden_dim'],
        time_dim=DIFFUSION_CONFIG['time_dim']
    ).to(DEVICE)
    
    checkpoint = torch.load(
        Path('checkpoints') / 'diffusion_best.pt',
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = DiffusionProcess(
        num_timesteps=DIFFUSION_CONFIG['num_timesteps'],
        beta_start=DIFFUSION_CONFIG['beta_start'],
        beta_end=DIFFUSION_CONFIG['beta_end']
    )
    
    # Generate samples
    with torch.no_grad():
        latent_samples = diffusion.sample(
            model,
            shape=(n_samples, DIFFUSION_CONFIG['latent_dim']),
            return_trajectory=False
        )
        
        images = vae.decode(latent_samples).cpu().numpy().reshape(n_samples, 28, 28)
    
    return images


def visualize_comparison(sindy_images, diffusion_images):
    """Create side-by-side comparison."""
    print("\nCreating comparison visualization...")
    
    n_samples = min(len(sindy_images), len(diffusion_images), 16)
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Generated Digits: SINDy vs Neural Diffusion', fontsize=16, y=0.98)
    
    for i in range(n_samples):
        row = i // 4
        col = (i % 4) * 2
        
        # SINDy generation
        axes[row, col].imshow(sindy_images[i], cmap='gray')
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title('SINDy', fontsize=10)
        
        # Diffusion generation
        axes[row, col + 1].imshow(diffusion_images[i], cmap='gray')
        axes[row, col + 1].axis('off')
        if row == 0:
            axes[row, col + 1].set_title('Neural', fontsize=10)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / 'sindy_vs_diffusion_generation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison to {save_path}")
    plt.show()


def visualize_trajectories(trajectories, n_show=3):
    """Visualize latent trajectories during generation."""
    print("\nVisualizing generation trajectories...")
    
    fig, axes = plt.subplots(n_show, 2, figsize=(12, 3*n_show))
    
    for i in range(n_show):
        traj = trajectories[i]  # (n_steps, 16)
        t = np.linspace(0, 1, traj.shape[0])
        
        # Plot first 2 dimensions
        axes[i, 0].plot(t, traj[:, 0], 'b-', linewidth=2, label='z_0')
        axes[i, 0].set_ylabel('z_0 value')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_title(f'Sample {i+1}: Dimension 0')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(t, traj[:, 1], 'r-', linewidth=2, label='z_1')
        axes[i, 1].set_ylabel('z_1 value')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_title(f'Sample {i+1}: Dimension 1')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / 'sindy_generation_trajectories.png'
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved trajectories to {save_path}")
    plt.show()


def main():
    """Main generation script."""
    print("="*80)
    print("GENERATING DIGITS WITH DISCOVERED SINDY EQUATIONS")
    print("="*80)
    
    # Load models
    sindy_model, vae = load_models()
    
    # Generate with SINDy
    sindy_images, sindy_trajs = generate_with_sindy(sindy_model, vae, n_samples=16)
    
    # Generate with diffusion for comparison
    try:
        diffusion_images = generate_with_diffusion(vae, None, None, n_samples=16)
    except Exception as e:
        print(f"\nNote: Could not generate diffusion samples for comparison: {e}")
        print("Showing SINDy results only...")
        diffusion_images = sindy_images  # Just duplicate for now
    
    # Visualize
    visualize_comparison(sindy_images, diffusion_images)
    visualize_trajectories(sindy_trajs, n_show=3)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print("\nResults:")
    print(f"  - Generated {len(sindy_images)} digits using SINDy equations")
    print(f"  - Images saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()