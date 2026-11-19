"""
Streamlit Control & Monitoring Dashboard for MEA Absorption Column
with LSTM Soft Sensor

Interactive UI simulating live plant disturbances and soft-sensor tracking.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from lstm_soft_sensor import create_model, MEASoftSensorLSTM
from rate_based_column import simulate_column_step_response, calculate_column_performance, reboiler_energy_duty
from synthetic_data import generate_synthetic_dataset, quick_inspection

# Page configuration
st.set_page_config(
    page_title="MEA CCUS Rate-Based Model + LSTM Soft Sensor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e2e6;
        border-radius: 0.5rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .disturbance-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .step-up { background: #ff4b4b; color: white; }
    .step-down { background: #2ecc71; color: white; }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# Initialize Session State
# ============================================================================

def init_session_state():
    """Initialize or reset session state variables."""
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = {'loss': [], 'val_loss': []}
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = {}


init_session_state()


# ============================================================================
# Model Functions
# ============================================================================

@st.cache_resource
def load_soft_sensor_model(input_dim=7, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.2):
    """Create and cache the LSTM soft sensor model."""
    model = MEASoftSensorLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    return model


def prepare_lstm_data(X, seq_len=10, feature_dim=7):
    """Prepare data sequences for LSTM model."""
    if len(X) <= seq_len:
        # Pad or repeat
        pad_len = seq_len - len(X)
        X_padded = np.pad(X, ((pad_len, 0), (0, 0)), mode='constant')
    else:
        X_padded = X[:seq_len]
    
    # Reshape for LSTM: (batch, seq_len, feature_dim)
    # For single sample inference
    X_reshaped = X_padded.reshape(1, seq_len, feature_dim)
    return X_reshaped


def predict_soft_sensor(model, X, seq_len=10, feature_dim=7):
    """Run LSTM soft sensor prediction."""
    model.eval()
    with torch.no_grad():
        X_seq = prepare_lstm_data(X, seq_len, feature_dim)
        # Convert to tensor if needed
        if not isinstance(X_seq, torch.Tensor):
            X_seq = torch.from_numpy(X_seq.astype(np.float32))
        
        prediction = model(X_seq)
        return prediction.numpy().flatten()


# ============================================================================
# Column Simulation
# ============================================================================

def run_simulation(
    duration=200,
    dt=1.0,
    disturbance='gas_velocity',
    magnitude=0.2,
    auto_reset=True
):
    """Run the rate-based column simulation."""
    with st.spinner("Simulating column dynamics..."):
        times, results = simulate_column_step_response(
            duration=duration,
            dt=dt,
            disturbance=disturbance,
            disturbance_magnitude=magnitude
        )
    
    st.session_state.simulation_results = results
    st.session_state.current_step = 0
    st.session_state.simulation_running = False
    
    return times, results


# ============================================================================
# Data Generation
# ============================================================================

def generate_training_data(n_samples=5, duration=200, dt=1.0):
    """Generate synthetic training datasets."""
    with st.spinner(f"Generating {n_samples} synthetic datasets..."):
        batch = generate_synthetic_data_batch(
            n_samples=n_samples,
            duration=duration,
            dt=dt
        )
    
    st.session_state.dataset_loaded = True
    st.session_state.simulation_results = {
        'datasets': batch,
        'count': n_samples
    }
    
    return batch


def generate_synthetic_data_batch(n_samples=5, duration=200, dt=1.0):
    """Generate batch of synthetic datasets."""
    batch = []
    for i in range(n_samples):
        # Vary disturbance type
        disturbance_types = ['gas_velocity', 'reboiler_duty', 'inlet_temp', 'none']
        dt_type = disturbance_types[i % len(disturbance_types)]
        
        dataset = generate_synthetic_dataset(
            duration=duration,
            dt=dt,
            include_disturbances=(dt_type != 'none'),
            seed=42 + i
        )
        dataset['disturbance_type'] = dt_type
        batch.append(dataset)
    
    return batch


# ============================================================================
# LSTM Training
# ============================================================================

def train_lstm_soft_sensor(
    X_train, y_train,
    input_dim=7, hidden_dim=64, output_dim=2,
    num_layers=2, learning_rate=0.001,
    epochs=50, batch_size=16, val_split=0.2
):
    """Train the LSTM soft sensor model."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    # Prepare data
    X = torch.from_numpy(X_train.astype(np.float32))
    y = torch.from_numpy(y_train.astype(np.float32))
    
    dataset = TensorDataset(X, y)
    split_idx = int(len(dataset) * (1 - val_split))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [split_idx, len(dataset) - split_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = MEASoftSensorLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    history = {'train_loss': [], 'val_loss': []}
    
    n_epochs = min(epochs, 100)  # cap for demo
    
    for epoch in range(n_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        
        # Train
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / len(train_loader)
        
        # Validate
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item() / len(val_loader)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Train Loss: {epoch_train_loss:.6f}, "
                  f"Val Loss: {epoch_val_loss:.6f}")
    
    st.session_state.model = model
    st.session_state.training_history = history
    
    return model, history


# ============================================================================
# Main Dashboard Layout
# ============================================================================

def main():
    """Main dashboard application."""
    st.title("🌡️ MEA CCUS Rate-Based Model + LSTM Soft Sensor")
    st.markdown("**Rate-Based Carbon Capture Column with LSTM Soft Sensor for Real-Time Process Control**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Control Panel")
        
        # Phase selector
        phase = st.radio(
            "Select Phase:",
            ["📊 Simulation", "🤖 LSTM Soft Sensor", "📈 Data Generation", "🔧 Column Performance"],
            help="Navigate between different project phases"
        )
        
        st.markdown("---")
        
        # Simulation parameters
        if phase in ["📊 Simulation", "🔧 Column Performance"]:
            st.subheader("Column Parameters")
            duration = st.slider("Simulation Duration (s)", 50, 500, 200)
            disturbance = st.selectbox(
                "Disturbance Type",
                ["None", "Gas Velocity", "Reboiler Duty", "Inlet Temperature"]
            ).lower().replace(" ", "_")
            magnitude = st.slider("Disturbance Magnitude", 0.05, 0.5, 0.2)
            
            if st.button("Run Simulation", type="primary"):
                run_simulation(
                    duration=duration,
                    disturbance=disturbance,
                    magnitude=magnitude
                )
        
        # LSTM parameters
        if phase in ["🤖 LSTM Soft Sensor", "📈 Data Generation"]:
            st.subheader("LSTM Parameters")
            input_dim = st.number_input("Input Dimension", 3, 10, 7)
            hidden_dim = st.number_hidden("Hidden Dimension", 32, 256, 64)
            num_layers = st.slider("LSTM Layers", 1, 4, 2)
            learning_rate = st.float_learning_rate("Learning Rate", 0.001, 0.01)
            epochs = st.slider("Training Epochs", 10, 200, 50)
        
        # Data generation
        if phase == "📈 Data Generation":
            st.subheader("Data Generation")
            n_datasets = st.number_input("Number of Datasets", 1, 50, 5)
            if st.button("Generate Training Data", type="primary"):
                generate_training_data(n_samples=n_datasets)
        
        # Column performance
        if phase == "🔧 Column Performance":
            st.subheader("Performance Metrics")
            m_solvent = st.solver("Solvent Flow (kg/s)", 0.5, 5.0, 1.0)
            if st.button("Calculate Performance"):
                perf = calculate_column_performance(m_solvent=m_solvent)
                st.session_state.performance_metrics = perf
    
    # Main content area
    if phase == "📊 Simulation":
        show_simulation_tab()
    elif phase == "🤖 LSTM Soft Sensor":
        show_lstm_tab()
    elif phase == "📈 Data Generation":
        show_data_generation_tab()
    elif phase == "🔧 Column Performance":
        show_performance_tab()


def show_simulation_tab():
    """Show the simulation tab content."""
    st.header("📊 Column Dynamics Simulation")
    
    if st.session_state.simulation_results:
        results = st.session_state.simulation_results
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_eff = results['capture_efficiency'][0]
            final_eff = results['capture_efficiency'][-1]
            delta_eff = final_eff - initial_eff
            st.metric(
                "Capture Efficiency",
                f"{final_eff:.1f}%",
                f"{delta_eff:+.1f}%"
            )
        with col2:
            st.metric("Simulation Steps", len(results['times']))
        with col3:
            total_duty = results['reboiler_duty']['total_duty']
            st.metric("Total Reboiler Duty", f"{total_duty:.1f} kW")
        
        # Time series plots
        fig = make_subplots(rows=3, cols=2, 
                           subplot_titles=[
                               "Gas-phase CO2 Concentration",
                               "Liquid-phase CO2 Concentration",
                               "Gas Temperature",
                               "Liquid Temperature",
                               "Capture Efficiency",
                               "Reboiler Duty"
                           ])
        
        fig.add_trace(
            go.Scatter(x=results['times'], y=results['C_G_CO2'], 
                      name="C_G,CO2", line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=results['times'], y=results['C_L_CO2'], 
                      name="C_L,CO2", line=dict(color='green')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=results['times'], y=results['T_gas'], 
                      name="T_gas", line=dict(color='red')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=results['times'], y=results['T_liquid'], 
                      name="T_liquid", line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=results['times'], y=results['capture_efficiency'], 
                      name="η_CO2", line=dict(color='purple')),
            row=3, col=1
        )
        
        # Reboiler duty as bar chart (simplified)
        fig.add_trace(
            go.Bar(x=results['times'], y=[total_duty/10]*len(results['times']), 
                  name="Q_reb (normalized)", marker_color='red', showlegend=False),
            row=3, col=2
        )
        
        fig.update_xaxis(title="Time (s)")
        fig.update_yaxis(title="Concentration (mol/m³)", row=1, col=1)
        fig.update_yaxis(title="Concentration (mol/m³)", row=1, col=2)
        fig.update_yaxis(title="Temperature (°C)", row=2, col=1)
        fig.update_yaxis(title="Temperature (°C)", row=2, col=2)
        fig.update_yaxis(title="Capture Efficiency (%)", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results expander
        with st.expander("View Detailed Results"):
            st.dataframe(pd.DataFrame({
                'Time (s)': results['times'],
                'C_G_CO2': results['C_G_CO2'],
                'C_L_CO2': results['C_L_CO2'],
                'T_Gas': results['T_gas'],
                'T_Liquid': results['T_liquid'],
                'Capture_Eff_%': results['capture_efficiency'],
            }))
    else:
        st.info("Click 'Run Simulation' in the sidebar to start a simulation.")
        st.markdown("""
        The simulation models a rate-based MEA absorption column with:
        - Gas-phase and liquid-phase mass balances
        - Two-film interfacial mass transfer
        - MEA reaction kinetics (carbamate/bicarbonate)
        - Reboiler energy duty calculation
        - Step-response disturbances
        """)


def show_lstm_tab():
    """Show the LSTM soft sensor tab."""
    st.header("🤖 LSTM Soft Sensor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Soft Sensor Configuration")
        
        # Model parameters
        input_dim = st.number_input("Input Dim", 3, 15, 7, key="lstm_input")
        hidden_dim = st.number_input("Hidden Dim", 16, 256, 64, key="lstm_hidden")
        num_layers = st.slider("LSTM Layers", 1, 4, 2, key="lstm_layers")
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, key="lstm_dropout")
        
        # Load model
        if st.button("Initialize Model", type="primary"):
            model = load_soft_sensor_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
            st.session_state.model = model
            st.session_state.model_initialized = True
            st.success(f"Model initialized: {input_dim}→{hidden_dim}→{num_layers} layers")
        
        if st.session_state.get('model_initialized', False) and st.session_state.model is not None:
            st.success("✅ Model ready for inference")
            
            # Make prediction
            if st.session_state.simulation_results:
                results = st.session_state.simulation_results
                X_sample = results['X']  # From synthetic data generation
                
                if X_sample.size > 0:
                    # Get a sample for prediction
                    sample_idx = st.slider("Select Time Step", 0, len(X_sample) - 1, 0)
                    X_sample_single = X_sample[sample_idx:sample_idx+1]
                    
                    model = st.session_state.model
                    prediction = predict_soft_sensor(model, X_sample_single.flatten())
                    
                    st.write("**Prediction Results:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "CO2 Capture Efficiency",
                            f"{prediction[0]:.1f}%"
                        )
                    with col_b:
                        st.metric(
                            "Overhead Purity",
                            f"{prediction[1]:.1f}%"
                        )
                    
                    # Show input features
                    st.write("Input Features (tray temperatures + flow):")
                    feature_names = ['T_1', 'T_3', 'T_5', 'T_7', 'T_10', 'Rich Flow', 'Flue Gas Temp']
                    for name, val in zip(feature_names, X_sample_single[0]):
                        st.write(f"  {name}: {val:.1f}")
    
    with col2:
        st.subheader("Training Progress")
        
        if st.session_state.get('training_history', {}).get('train_loss'):
            history = st.session_state.training_history
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['train_loss'],
                name='Train Loss',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                name='Val Loss',
                line=dict(color='red', dash='dot')
            ))
            fig.update_layout(
                title="Training Loss",
                yaxis_title="MSE Loss",
                xaxis_title="Epoch",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train a model to see progress")
            if st.button("Generate Dummy Training Data"):
                np.random.seed(42)
                X_dummy = np.random.randn(100, 50, input_dim)
                y_dummy = np.random.randn(100, 2)
                
                with st.spinner("Training model..."):
                    model, history = train_lstm_soft_sensor(
                        X_dummy, y_dummy,
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        learning_rate=learning_rate,
                        epochs=epochs
                    )
                
                st.session_state.training_history = history
                st.session_state.model = model
                st.rerun()
        
        # Model info
        if st.session_state.get('model_initialized', False):
            st.markdown("---")
            st.caption("Model Architecture:")
            m = st.session_state.model
            st.caption(f"LSTM({input_dim}→{hidden_dim}, {num_layers} layers)")
            st.caption(f"Dropout: {dropout}")
            st.caption(f"Output: {input_dim} → Linear → 2 outputs")


def show_data_generation_tab():
    """Show the data generation tab."""
    st.header("📈 Synthetic Data Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generate Training Datasets")
        
        n_samples = st.number_input("Number of Datasets", 1, 20, 3)
        duration = st.slider("Duration (s)", 50, 500, 200)
        dt = st.slider("Time Step (s)", 0.1, 5.0, 1.0)
        
        if st.button("Generate Data", type="primary"):
            batch = generate_training_data(n_samples=n_samples, duration=duration, dt=dt)
            
            # Show inspection for first dataset
            if batch:
                quick_inspection(batch[0])
                
                # Save sample
                df = save_dataset(batch[0], 'synthetic_data_sample.csv')
                st.success(f"Generated {n_samples} datasets. Sample saved.")
    
    with col2:
        st.subheader("Dataset Inspection")
        
        if st.session_state.get('dataset_loaded', False) and 'datasets' in st.session_state.simulation_results:
            datasets = st.session_state.simulation_results['datasets']
            st.write(f"Loaded {len(datasets)} datasets")
            
            # Select dataset to inspect
            ds_idx = st.selectbox("Select Dataset", range(len(datasets)), 
                                  format_func=lambda i: f"Dataset {i+1} ({datasets[i]['disturbance_type']})")
            
            if st.button("Inspect Selected"):
                quick_inspection(datasets[ds_idx])
        else:
            st.info("Generate data first to inspect")
            if st.button("Use Sample Data"):
                # Load sample if exists
                import os
                if os.path.exists('synthetic_data_sample.csv'):
                    df = pd.read_csv('synthetic_data_sample.csv')
                    st.write("Sample dataset preview:")
                    st.dataframe(df.head())
                else:
                    st.warning("No sample data found. Generate data first.")


def show_performance_tab():
    """Show the column performance tab."""
    st.header("🔧 Column Performance Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Performance Calculator")
        m_solvent = st.slider(
            "Solvent Flow Rate (kg/s)",
            0.5, 5.0, 1.0, 0.1,
            help="Total aqueous MEA solvent circulation rate"
        )
        
        T_lean = st.slider(
            "Lean Solvent Temperature (°C)",
            30, 80, 40,
            help="Temperature of solvent entering absorption column"
        )
        
        if st.button("Calculate Performance", type="primary"):
            perf = calculate_column_performance(m_solvent=m_solvent)
            st.session_state.performance_metrics = perf
    
    with col2:
        st.subheader("Performance Results")
        
        if 'performance_metrics' in st.session_state:
            perf = st.session_state.performance_metrics
            
            st.metric("Lean Loading", f"{perf['lean_loading']:.3f}")
            st.metric("Rich Loading", f"{perf['rich_loading']:.3f}")
            st.metric("CO2 Removal", f"{perf['co2_removal_mol_s']:.3f} mol/s")
            st.metric("Reboiler Duty", f"{perf['reboiler_duty_kw']:.1f} kW")
            st.metric("Specific Energy", f"{perf['specific_energy_kwh_kgco2']:.1f} kWh/kg CO₂")
            st.metric("Approach Temperature", f"{perf['approach_temp']:.1f} K")
            
            # Economic indicator
            st.caption(f"\\n**Energy Penalty**: {perf['specific_energy_kwh_kgco2']:.1f} kWh per kg CO₂ captured")
            st.caption(f"\\n**Typical Benchmark**: 3.5–5.0 kWh/kg CO₂ for MEA process")
        else:
            st.info("Adjust parameters and click 'Calculate Performance'")


if __name__ == "__main__":
    main()