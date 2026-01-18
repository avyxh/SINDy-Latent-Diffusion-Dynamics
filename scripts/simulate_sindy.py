import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from config import RESULTS_DIR, TRAJECTORY_CONFIG

def exp_decay(x):
    return np.exp(-np.abs(x))

def gaussian(x):
    return np.exp(-x**2)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def name_exp(x): return f"exp(-|{x}|)"
def name_gauss(x): return f"exp(-{x}^2)"
def name_sigmoid(x): return f"sigmoid({x})"

def simulate_discovered_system():
    # 1. Load the SINDy model
    with open(RESULTS_DIR / 'sindy_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 2. Load original trajectories for comparison (initial conditions)
    data = np.load(TRAJECTORY_CONFIG['save_path'], allow_pickle=True)
    orig_trajs = data['trajectories'] # (n_traj, n_steps, n_dims)
    
    # Select a random trajectory to "reconstruct"
    traj_idx = 0
    z0 = orig_trajs[traj_idx, 0, :]
    t_span = (0, 1) # Normalized time used in SINDy training
    t_eval = np.linspace(0, 1, orig_trajs.shape[1])

    # 3. Define the ODE function based on SINDy
    def sindy_ode(t, z):
        # SINDy expects [z_0, ..., z_n, t] as input
        z_augmented = np.concatenate([z, [t]]).reshape(1, -1)
        dzdt = model.predict(z_augmented)
        return dzdt.flatten()

    # 4. Integrate
    print(f"Simulating reconstruction for trajectory {traj_idx}...")
    sol = solve_ivp(sindy_ode, t_span, z0, t_eval=t_eval, method='RK45')

    # 5. Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot first 2 dimensions
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.plot(t_eval, orig_trajs[traj_idx, :, i], 'k--', alpha=0.6, label='Original (Diffusion)')
        plt.plot(sol.t, sol.y[i, :], 'r-', linewidth=2, label='Reconstructed (SINDy)')
        plt.title(f'Dimension z_{i}')
        plt.xlabel('Time (normalized)')
        plt.ylabel('Latent Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'sindy_reconstruction.png')
    plt.show()
    
    print(f"✓ Simulation complete! Plot saved to {RESULTS_DIR}")

if __name__ == "__main__":
    simulate_discovered_system()