"""
Synthetic Data Generation for MEA Absorption Column

Generates high-resolution temporal training datasets 
(Temperature profiles vs. Purity) for LSTM soft sensor training.
"""

import numpy as np
import pandas as pd


# ============================================================================
# Synthetic Plant Model for Data Generation
# ============================================================================

# Baseline operating parameters
BASELINE_PARAMS = {
    'u_G': 1.5,        # gas velocity (m/s)
    'u_L': 0.02,       # liquid velocity (m/s)
    'eps_G': 0.85,     # gas holdup
    'eps_L': 0.15,     # liquid holdup
    'a_i': 250.0,      # interfacial area (m²/m³)
    'C_MEA': 3000.0,   # MEA concentration (mol/m³)
    'C_CO2_loaded_lean': 200.0,  # lean solvent loading (mol/m³)
    'C_CO2_loaded_rich': 500.0,  # rich solvent loading (mol/m³)
    'UA': 500.0,       # heat transfer coefficient (W/m·K)
    'T_ambient': 25.0, # ambient temperature (°C)
    'T_INLET': 40.0,   # flue gas inlet temperature (°C)
}


# ============================================================================
# Synthetic Data Generation Functions
# ============================================================================

def generate_synthetic_dataset(
    duration=200.0,       # seconds
    dt=1.0,               # time step (seconds)
    include_disturbances=True,
    seed=42
):
    """
    Generate synthetic training dataset for MEA column + LSTM soft sensor.

    Returns:
        dict with keys:
        - 'times': time array (n_points,)
        - 'X': feature matrix (n_points, n_features)
        - 'y': target vector (n_points, output_dim=2)
        - 'capture_efficiency': CO2 capture efficiency (%)
        - 'overhead_purity': CO2 purity in overhead (%)
        - 'reboiler_duty': reboiler energy duty profile (kW)
        - 'tray_temperatures': 5 sensor temperatures profile
        - 'disturbances': dict of disturbance profiles
    """
    np.random.seed(seed)

    # Time grid
    n_points = int(duration / dt) + 1
    times = np.linspace(0, duration, n_points)

    # === Generate Disturbance Profiles ===
    # Simulate plant disturbances over time

    # Gas velocity profile (baseline with possible step change)
    if include_disturbances:
        # Base gas velocity with 20% step increase at t=50s + sinusoidal variation
        u_G_profile = BASELINE_PARAMS['u_G'] * np.ones(n_points)
        step_idx = int(50.0 / dt)
        if step_idx < n_points:
            u_G_profile[step_idx:] *= 1.2
        # Add sinusoidal variation (slow cycle)
        u_G_profile += 0.05 * np.sin(2 * np.pi * 0.005 * times)
        # Add some random walk
        u_G_profile += np.cumsum(np.random.normal(0, 0.01, n_points)) * dt
    else:
        u_G_profile = np.full(n_points, BASELINE_PARAMS['u_G'])

    # Reboiler duty profile (baseline with possible step changes)
    if include_disturbances:
        Q_reb_profile = np.full(n_points, 2500.0)  # kW baseline
        # Step changes at t=80s and t=150s
        idx_80 = int(80.0 / dt)
        idx_150 = int(150.0 / dt)
        if idx_80 < n_points:
            Q_reb_profile[idx_80:] = 2700.0
        if idx_150 < n_points:
            Q_reb_profile[idx_150:] = 2600.0
        # Add noise
        Q_reb_profile += np.random.normal(0, 50, n_points)
    else:
        Q_reb_profile = np.full(n_points, 2500.0)

    # === Generate Response Profiles ===
    # Simulated column responses to disturbances

    # CO2 concentration in gas phase (decreases with capture, increases with disturbances)
    # Baseline capture efficiency ~90%
    baseline_efficiency = 90.0

    # Gas-phase CO2 (normalized)
    C_G_CO2 = np.ones(n_points) * (u_G_profile / BASELINE_PARAMS['u_G']) * (1 - baseline_efficiency / 100.0)

    # Apply some dynamics: first-order lag to simulate column response
    tau = 30.0  # time constant ~30s
    for i in range(1, n_points):
        # Simple exponential filter
        C_G_CO2[i] = C_G_CO2[i-1] + (C_G_CO2[i] - C_G_CO2[i-1]) * dt / tau

    # Liquid-phase loading (alpha, mol CO2 / mol MEA)  (Formulation B)
    # alpha ranges from ~0.05 (lean) to ~0.2 (rich) in typical MEA service
    alpha = 0.05 + 0.15 * np.clip(1.0 - C_G_CO2 / 8.0, 0, 1)

    # Gas temperature (flue gas)
    T_gas = BASELINE_PARAMS['T_INLET'] + 5.0 * np.sin(2 * np.pi * 0.01 * times) + \
            2.0 * np.random.normal(0, 1, n_points)

    # Liquid temperature (column internals)
    T_liquid = 40.0 + 25.0 * (1 - np.exp(-times / 60.0)) + \
               3.0 * np.sin(2 * np.pi * 0.008 * times) + \
               1.0 * np.random.normal(0, 1, n_points)

    # === Calculate Derived Quantities ===

    # Capture efficiency (varies with disturbances)
    # When gas velocity increases, capture typically decreases
    efficiency_adjustment = 1.0 - 0.5 * (u_G_profile - BASELINE_PARAMS['u_G']) / BASELINE_PARAMS['u_G']
    capture_efficiency = np.clip(90.0 + 5.0 * efficiency_adjustment + \
                                  3.0 * np.sin(2 * np.pi * 0.005 * times) + \
                                  2.0 * np.random.normal(0, 1, n_points), 0, 100)

    # Overhead purity (CO2 in clean air = 100 - slippage)
    # Slippage inversely related to capture efficiency
    CO2_slippage = 100.0 - capture_efficiency + np.random.normal(0, 2, n_points)
    overhead_purity = np.clip(CO2_slippage, 0, 100)

    # Reboiler duty profile (already generated above)
    reboiler_duty = Q_reb_profile

    # === Generate Sparse Temperature Sensors ===
    # 5 temperature sensors across column trays (map to tray locations)

    # Bottom trays are cooler, top trays are hotter
    # Tray 1 (bottom) - near liquid inlet
    # Tray 3 (middle-lower)
    # Tray 5 (mid-column)
    # Tray 7 (middle-upper)
    # Tray 10 (top) - near reboiler/reflux

    # Map based on liquid temperature profile
    # Normalize time to get tray positions
    tray_positions = np.linspace(0, len(T_liquid) - 1, 5).astype(int)

    # Get temperatures at tray positions
    tray_temperatures = T_liquid[tray_positions] if len(T_liquid) >= 5 else T_liquid

    # Add measurement noise (±0.5°C)
    sensor_noise = np.random.normal(0, 0.5, 5)
    tray_temperatures_noisy = tray_temperatures + sensor_noise

    # Also include gas temperature proxy and flow rate
    rich_amine_flow = np.mean(u_G_profile) * 10  # proxy for flow
    flue_gas_temp = np.mean(T_gas)

    # === Create Feature Matrix for LSTM ===
    # Input features per time step:
    # [T_1, T_3, T_5, T_7, T_10, rich_amine_flow, flue_gas_temp]

    n_features = 7  # 5 tray temps + flow + flue gas temp
    X_features = np.zeros((n_points, n_features))

    # Ensure we have 5 tray temperatures even if profile is short
    if len(tray_temperatures_noisy) >= 5:
        tray_vals = tray_temperatures_noisy[:5]
    else:
        # Repeat last value if not enough
        tray_vals = np.tile(tray_temperatures_noisy[-1] if len(tray_temperatures_noisy) > 0 else 40.0, 5)

    for i in range(n_points):
        # Interpolate tray temperatures at each time step
        # For simplicity, use the same tray temperatures (they change slowly)
        t_idx = min(i, len(tray_vals) - 1)
        X_features[i] = [
            tray_vals[t_idx],       # T_1 (bottom tray)
            tray_vals[min(t_idx+1, len(tray_vals)-1)],  # T_3
            tray_vals[min(t_idx+2, len(tray_vals)-1)],  # T_5 (mid-column)
            tray_vals[min(t_idx+3, len(tray_vals)-1)],  # T_7
            tray_vals[min(t_idx+4, len(tray_vals)-1)],  # T_10 (top tray)
            rich_amine_flow,        # rich amine feed flow rate
            flue_gas_temp           # flue gas inlet temperature
        ]

    # === Target Variables ===
    # Output: [capture_efficiency, overhead_purity]
    y_targets = np.column_stack([
        capture_efficiency,
        overhead_purity
    ])

    dataset = {
        'times': times,
        'X': X_features,           # (n_points, seq_len, input_dim) - will be reshaped for LSTM
        'y': y_targets,            # (n_points, output_dim)
        'capture_efficiency': capture_efficiency,
        'overhead_purity': overhead_purity,
        'reboiler_duty': reboiler_duty,
        'tray_temperatures': tray_temperatures_noisy,
        'T_gas': T_gas,
        'T_liquid': T_liquid,
        'u_G_profile': u_G_profile,
        'Q_reb_profile': reboiler_duty,
        'params': BASELINE_PARAMS.copy(),
        'disturbances': {
            'gas_velocity_profile': u_G_profile,
            'reboiler_duty_profile': reboiler_duty
        }
    }

    return dataset


def generate_training_batch(
    n_samples=100,
    duration=200.0,
    dt=1.0,
    disturbance_types=None,
    seed=None
):
    """
    Generate a batch of synthetic datasets with varied disturbance patterns.

    Args:
        n_samples: number of datasets to generate
        duration: simulation duration per sample
        dt: time step
        disturbance_types: list of disturbance types to include
        seed: random seed (or list of seeds, one per sample)

    Returns:
        list of dataset dicts
    """
    if seed is None:
        seeds = [np.random.randint(0, 10000) for _ in range(n_samples)]
    elif isinstance(seed, int):
        seeds = [seed + i for i in range(n_samples)]
    else:
        seeds = seed

    batch = []
    for i, s in enumerate(seeds):
        # Vary disturbance type
        if disturbance_types is None:
            disturbance_types_i = ['gas_velocity', 'reboiler_duty', 'inlet_temp', 'none']
            dt_disturb = disturbance_types_i[i % len(disturbance_types_i)]
        else:
            dt_disturb = disturbance_types[i % len(disturbance_types)]

        dataset = generate_synthetic_dataset(
            duration=duration,
            dt=dt,
            include_disturbances=(dt_disturb != 'none'),
            seed=s
        )
        dataset['disturbance_type'] = dt_disturb
        batch.append(dataset)

    return batch


# ============================================================================
# Data Export & Exploration Functions
# ============================================================================

def save_dataset(dataset, filepath):
    """Save dataset to CSV for inspection (first few hundred points)."""
    times = dataset['times']
    n_points = len(times)

    # Ensure reboiler_duty is an array of correct length
    reboiler_duty = dataset['reboiler_duty']
    if np.isscalar(reboiler_duty):
        reboiler_duty = np.full(n_points, reboiler_duty)

    # Ensure capture_efficiency and overhead_purity are arrays
    capture_eff = dataset['capture_efficiency']
    overhead_purity = dataset['overhead_purity']
    if np.isscalar(capture_eff):
        capture_eff = np.full(n_points, capture_eff)
    if np.isscalar(overhead_purity):
        overhead_purity = np.full(n_points, overhead_purity)

    # Create DataFrame with key columns - ensure all arrays have same length
    df = pd.DataFrame({
        'time_s': dataset['times'],
        'T_1_C': dataset['X'][:, 0],
        'T_3_C': dataset['X'][:, 1],
        'T_5_C': dataset['X'][:, 2],
        'T_7_C': dataset['X'][:, 3],
        'T_10_C': dataset['X'][:, 4],
        'rich_amine_flow': dataset['X'][:, 5],
        'flue_gas_temp_C': dataset['X'][:, 6],
        'capture_eff_%': capture_eff,
        'overhead_purity_%': overhead_purity,
        'reboiler_duty_kW': reboiler_duty,
    })

    # Verify all columns have same length
    lengths = [len(df[col]) for col in df.columns]
    if len(set(lengths)) != 1:
        # Truncate to minimum length
        min_len = min(lengths)
        print(f"Warning: Arrays have inconsistent lengths, truncating to {min_len}")
        for col in df.columns:
            df[col] = df[col].values[:min_len]

    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    print(f"Shape: {dataset['X'].shape} features, {dataset['y'].shape} targets")
    return df


def quick_inspection(dataset):
    """Quick statistical inspection of generated dataset."""
    print("=== Dataset Inspection ===")
    print(f"Times: {dataset['times'][0]:.1f} to {dataset['times'][-1]:.1f} s")
    print(f"Feature shape: {dataset['X'].shape}")
    print(f"Target shape: {dataset['y'].shape}")
    print(f"\nFeature ranges:")
    for i in range(dataset['X'].shape[1]):
        col_min = dataset['X'][:, i].min()
        col_max = dataset['X'][:, i].max()
        print(f"  Col {i}: [{col_min:.2f}, {col_max:.2f}]")
    print(f"\nTarget ranges:")
    for i in range(dataset['y'].shape[1]):
        col_min = dataset['y'][:, i].min()
        col_max = dataset['y'][:, i].max()
        print(f"  Col {i}: [{col_min:.2f}, {col_max:.2f}]")
    print(f"\nDisturbance: {dataset['disturbances']}")


if __name__ == "__main__":
    # Quick test: generate and inspect a dataset
    print("=== Synthetic Data Generation Test ===\n")

    dataset = generate_synthetic_dataset(
        duration=150.0,
        dt=1.0,
        include_disturbances=True,
        seed=123
    )

    quick_inspection(dataset)

    # Save for later use
    df = save_dataset(dataset, 'synthetic_data_sample.csv')

    # Print some sample data
    print(f"\nFirst 5 time points:")
    print(df.head())
    print(f"\nLast 5 time points:")
    print(df.tail())

    print("\n=== Test Complete ===")