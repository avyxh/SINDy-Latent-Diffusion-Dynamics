"""
Configuration file for Diffusion-SINDy project.
All hyperparameters and paths are defined here for easy modification.
"""
import torch
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================================
# VAE HYPERPARAMETERS
# ============================================================================
VAE_CONFIG = {
    "input_dim": 784,  # 28x28 MNIST images flattened
    "hidden_dim": 256,  # Hidden layer size
    "latent_dim": 16,   # Latent space dimension (key parameter!)
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 20,
    "beta": 1.0,  # Weight for KL divergence term
}

# ============================================================================
# DIFFUSION MODEL HYPERPARAMETERS
# ============================================================================
DIFFUSION_CONFIG = {
    "latent_dim": 16,  # Must match VAE latent_dim
    "hidden_dim": 128,  # Hidden dimensions in U-Net-like model
    "num_timesteps": 50,  # Total diffusion steps
    "beta_start": 1e-4,   # Starting noise schedule
    "beta_end": 0.02,     # Ending noise schedule
    "learning_rate": 1e-4,
    "batch_size": 128,
    "epochs": 30,
}

# ============================================================================
# TRAJECTORY COLLECTION
# ============================================================================
TRAJECTORY_CONFIG = {
    "num_trajectories": 100,  # Number of denoising trajectories to collect
    "trajectories_per_class": 10,  # 10 per digit (0-9)
    "save_path": RESULTS_DIR / "trajectories.npz",
}

# ============================================================================
# SINDY HYPERPARAMETERS
# ============================================================================
SINDY_CONFIG = {
    'poly_order': 3, 
    'threshold': 0.1,  
    'alpha': 0.01,  # Lower regularization
    'max_iter': 100,  # More iterations for convergence
    'include_bias': True,
}

# ============================================================================
# RANDOM SEEDS (for reproducibility)
# ============================================================================
SEED = 42

def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)