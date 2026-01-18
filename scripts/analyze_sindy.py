"""
Step 4: Analyze trajectories with SINDy to discover governing equations.

This is the core of the project: apply sparse dynamics discovery to
reveal interpretable mathematical structure in diffusion generation.

Usage:
    python scripts/4_analyze_sindy.py
"""
import sys

import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import TRAJECTORY_CONFIG, DIFFUSION_CONFIG, SINDY_CONFIG, RESULTS_DIR
from utils import plot_sindy_coefficients

    # Custom functions (exponentials for diffusion physics)
def exp_decay(x):
    return np.exp(-np.abs(x))

def gaussian(x):
    return np.exp(-x**2)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def name_exp(x): return f"exp(-|{x}|)"
def name_gauss(x): return f"exp(-{x}^2)"
def name_sigmoid(x): return f"sigmoid({x})"


def compute_velocities(trajectories, dt=1.0):
    """
    Compute velocities from trajectories using finite differences.
    
    Args:
        trajectories: Array (n_traj, n_steps, n_dims)
        dt: Time step size
        
    Returns:
        positions: Array (n_samples, n_dims)
        velocities: Array (n_samples, n_dims)
        timesteps: Array (n_samples,)
    """
    n_traj, n_steps, n_dims = trajectories.shape
    
    # Use central differences for interior points
    # Skip first and last timestep
    velocities = (trajectories[:, 2:, :] - trajectories[:, :-2, :]) / (2 * dt)
    positions = trajectories[:, 1:-1, :]
    
    # Flatten across trajectories
    positions_flat = positions.reshape(-1, n_dims)
    velocities_flat = velocities.reshape(-1, n_dims)
    
    # Create timestep array (normalized to [0, 1])
    timesteps = np.tile(np.arange(1, n_steps-1), n_traj) / (n_steps - 1)
    
    return positions_flat, velocities_flat, timesteps


def build_library_with_time(latent_dim, poly_order=2):
    """
    Build SINDy library including time-dependent terms.
    
    This is crucial for diffusion dynamics where time-dependence matters!
    
    Args:
        latent_dim: Number of latent dimensions
        poly_order: Maximum polynomial order
        
    Returns:
        feature_library: PySINDy library object
    """
    # Create standard polynomial library
    poly_lib = ps.PolynomialLibrary(
        degree=poly_order,
        include_bias=SINDY_CONFIG['include_bias']
    )
    
    # For time-dependent terms, we'll include time as a "state variable"
    # This allows terms like t, t*z_i, etc.
    
    return poly_lib


def analyze_with_sindy(positions, velocities, timesteps):
    """
    Apply SINDy to discover governing equations.
    
    Args:
        positions: State snapshots (n_samples, n_dims)
        velocities: Time derivatives (n_samples, n_dims)
        timesteps: Time values (n_samples,)
        
    Returns:
        model: Fitted SINDy model
        feature_names: Names of library functions
    """
    n_samples, n_dims = positions.shape
    
    print("\nApplying SINDy...")
    print(f"  Data shape: {positions.shape}")
    print(f"  Polynomial order: {SINDY_CONFIG['poly_order']}")
    print(f"  Sparsity threshold: {SINDY_CONFIG['threshold']}")
    
    # Augment state with time as additional "dimension"
    # This allows discovering time-dependent terms
    X_augmented = np.column_stack([positions, timesteps[:, None]])
    
    # Create feature names
    state_names = [f'z_{i}' for i in range(n_dims)] + ['t']
    
    # Build library
    poly_lib = ps.PolynomialLibrary(
        degree=3,
        include_bias=True
    )
    
    # 2. Fourier terms (for periodic structure)
    fourier_lib = ps.FourierLibrary(
        n_frequencies=2,
        include_sin=True,
        include_cos=True
    )
    
    custom_lib = ps.CustomLibrary(
        library_functions=[exp_decay, gaussian, sigmoid],
        function_names=[name_exp, name_gauss, name_sigmoid]
    )
    
    # Combine all libraries
    library = poly_lib + fourier_lib + custom_lib
    
    # Create optimizer (STLSQ - Sequential Thresholded Least Squares)
    optimizer = ps.STLSQ(
        threshold=SINDY_CONFIG['threshold'],
        alpha=SINDY_CONFIG['alpha'],
        max_iter=SINDY_CONFIG['max_iter'],
        normalize_columns=True
    )
    
    # Create SINDy model
    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        feature_names=state_names
    )
    
    # Fit model
    # Note: We fit velocities excluding the time dimension
    model.fit(X_augmented, x_dot=velocities, t=1.0)
    
    print("\n✓ SINDy fitting complete!")
    
    return model, state_names


def analyze_discovered_equations(model, state_names):
    """
    Analyze and interpret discovered equations.
    
    Args:
        model: Fitted SINDy model
        state_names: Names of state variables
    """
    print("\n" + "="*80)
    print("DISCOVERED EQUATIONS")
    print("="*80)
    
    # Print equations
    model.print()
    
    # Get coefficients
    coefficients = model.coefficients()
    feature_names = model.get_feature_names()
    
    print("\n" + "="*80)
    print("COEFFICIENT ANALYSIS")
    print("="*80)
    
    n_dims = coefficients.shape[0]
    
    for dim in range(n_dims):
        coef = coefficients[dim, :]
        nonzero_mask = np.abs(coef) > 1e-6
        n_nonzero = nonzero_mask.sum()
        
        print(f"\nDimension z_{dim}:")
        print(f"  Active terms: {n_nonzero}/{len(feature_names)}")
        print(f"  Sparsity: {100 * (1 - n_nonzero/len(feature_names)):.1f}%")
        
        if n_nonzero > 0:
            # Show top 5 terms by magnitude
            top_indices = np.argsort(np.abs(coef))[-5:][::-1]
            print(f"  Top terms:")
            for idx in top_indices:
                if np.abs(coef[idx]) > 1e-6:
                    print(f"    {coef[idx]:+.4f} × {feature_names[idx]}")
    
    # Overall sparsity
    total_terms = coefficients.size
    nonzero_terms = (np.abs(coefficients) > 1e-6).sum()
    overall_sparsity = 100 * (1 - nonzero_terms / total_terms)
    
    print(f"\nOverall Statistics:")
    print(f"  Total possible terms: {total_terms}")
    print(f"  Active terms: {nonzero_terms}")
    print(f"  Overall sparsity: {overall_sparsity:.1f}%")
    
    return coefficients, feature_names


def identify_patterns(coefficients, feature_names, latent_dim):
    """
    Identify interesting patterns in discovered equations.
    
    Args:
        coefficients: SINDy coefficient matrix
        feature_names: Names of features
        latent_dim: Number of latent dimensions (excluding time)
    """
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    # 1. Time-dependence analysis
    print("\n1. Time-Dependent vs Autonomous Terms:")
    time_dependent_terms = [i for i, name in enumerate(feature_names) if 't' in name]
    autonomous_terms = [i for i, name in enumerate(feature_names) if 't' not in name]
    
    for dim in range(min(latent_dim, coefficients.shape[1])):
        time_strength = np.abs(coefficients[dim, time_dependent_terms]).sum()
        auto_strength = np.abs(coefficients[dim, autonomous_terms]).sum()
        
        if auto_strength > 0:
            ratio = time_strength / auto_strength
        else:
            ratio = np.inf if time_strength > 0 else 0
            
        print(f"  z_{dim}: Time/Auto ratio = {ratio:.2f}", end="")
        if ratio > 2:
            print(" (strongly time-dependent)")
        elif ratio > 0.5:
            print(" (mixed)")
        else:
            print(" (mostly autonomous)")
    
    # 2. Coupling analysis
    print("\n2. Dimensional Coupling:")
    for dim in range(latent_dim):
        # Find which other dimensions appear in equation for this dimension
        coupled_dims = set()
        for idx, name in enumerate(feature_names):
            if np.abs(coefficients[dim, idx]) > 1e-6:
                # Extract dimension indices from feature name
                for other_dim in range(latent_dim):
                    if f'z_{other_dim}' in name and other_dim != dim:
                        coupled_dims.add(other_dim)
        
        if coupled_dims:
            print(f"  z_{dim} couples to: {sorted(coupled_dims)}")
        else:
            print(f"  z_{dim} evolves independently")
    
    # 3. Linear vs Nonlinear
    print("\n3. Linearity Analysis:")
    for dim in range(latent_dim):
        linear_terms = [i for i, name in enumerate(feature_names) 
                       if name.count('z_') == 1 and '^' not in name]
        nonlinear_terms = [i for i, name in enumerate(feature_names)
                          if name.count('z_') > 1 or '^' in name]
        
        linear_strength = np.abs(coefficients[dim, linear_terms]).sum()
        nonlinear_strength = np.abs(coefficients[dim, nonlinear_terms]).sum()
        
        print(f"  z_{dim}: Linear={linear_strength:.3f}, Nonlinear={nonlinear_strength:.3f}")


def main():
    """Main SINDy analysis."""
    print("="*80)
    print("STEP 4: SINDy Analysis of Diffusion Dynamics")
    print("="*80)
    
    # Load trajectories
    print("\nLoading trajectories...")
    data = np.load(TRAJECTORY_CONFIG['save_path'], allow_pickle=True)
    trajectories = data['trajectories']
    config = data['config'].item()
    
    print(f"  Trajectories shape: {trajectories.shape}")
    print(f"  Number of trajectories: {config['num_trajectories']}")
    print(f"  Timesteps per trajectory: {config['num_timesteps'] + 1}")
    print(f"  Latent dimension: {config['latent_dim']}")
    
    # Compute velocities
    print("\nComputing velocities...")
    positions, velocities, timesteps = compute_velocities(trajectories)
    
    print(f"  Positions shape: {positions.shape}")
    print(f"  Velocities shape: {velocities.shape}")
    print(f"  Timesteps shape: {timesteps.shape}")
    
    # Apply SINDy
    model, state_names = analyze_with_sindy(positions, velocities, timesteps)
    
    # Analyze results
    coefficients, feature_names = analyze_discovered_equations(model, state_names)
    
    # Identify patterns
    actual_latent_dim = coefficients.shape[0]
    identify_patterns(coefficients, feature_names, actual_latent_dim)
    """
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Plot coefficient heatmap
    plot_sindy_coefficients(
        coefficients,
        feature_names,
        latent_dim_idx='all',
        save_path=RESULTS_DIR / 'sindy_coefficients_heatmap.png'
    )
    
    # Plot individual dimensions
    for dim in range(min(3, config['latent_dim'])):
        plot_sindy_coefficients(
            coefficients,
            feature_names,
            latent_dim_idx=dim,
            save_path=RESULTS_DIR / f'sindy_coefficients_z{dim}.png'
        )
    """
    
    # Prediction accuracy
    print("\n" + "="*80)
    print("PREDICTION ACCURACY")
    print("="*80)
    
    # Predict velocities
    X_augmented = np.column_stack([positions, timesteps[:, None]])
    velocities_pred = model.predict(X_augmented)
    
    # Compute R² score for each dimension
    from sklearn.metrics import r2_score
    
    print("\nR² scores by dimension:")
    for dim in range(config['latent_dim']):
        r2 = r2_score(velocities[:, dim], velocities_pred[:, dim])
        print(f"  z_{dim}: R² = {r2:.4f}")
    
    overall_r2 = r2_score(velocities.flatten(), velocities_pred.flatten())
    print(f"\nOverall R²: {overall_r2:.4f}")
    
    # Save model
    model_path = RESULTS_DIR / 'sindy_model.pkl'
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Saved SINDy model to: {model_path}")
    
    print("\n" + "="*80)
    print("SINDy ANALYSIS COMPLETE!")
    print("="*80)
    print("\n✓ All analysis completed successfully!")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nKey findings:")
    print(f"  - Overall sparsity: {100 * (1 - (np.abs(coefficients) > 1e-6).sum() / coefficients.size):.1f}%")
    print(f"  - Overall R²: {overall_r2:.4f}")
    print(f"  - Active terms: {(np.abs(coefficients) > 1e-6).sum()} / {coefficients.size}")


if __name__ == '__main__':
    main()