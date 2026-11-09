# Rate-Based MEA Absorption Dynamic Model with LSTM Soft Sensor

This project implements a rate-based model for post-combustion carbon capture using Monoethanolamine (MEA) absorption, coupled with an LSTM soft sensor for real-time process control.

## Overview

### Architecture
1. **Rate-Based Column Model** - Solves differential mass and energy balance equations along the column height using film theory and interfacial kinetics
2. **LSTM Soft Sensor** - PyTorch LSTM network that predicts CO2 capture efficiency and purity from sparse temperature sensor data
3. **Streamlit Dashboard** - Interactive UI for simulating plant disturbances and monitoring soft-sensor tracking

### Mathematical Formulation

#### Governing Conservation Equations (Differential Segment along Column Height z):

1. **Gas-Phase Component Mass Balance:**
   ∂C_G,i/∂t = -(u_G/ε_G) ∂C_G,i/∂z - (a_i/ε_G) N_i

2. **Liquid-Phase Component Mass Balance:**
   ∂C_L,i/∂t = +(u_L/ε_L) ∂C_L,i/∂z + (a_i/ε_L) N_i + R_i

3. **Interfacial Mass Transfer Flux (Two-Film Theory):**
   N_CO2 = K_overall (C_G,CO2 - m · C_L,CO2*)

4. **Reboiler Thermal Energy Duty:**
   Q_reb = m_solvent · Cp · (T_strip,out - T_strip,in) + ΔH_des · m_CO2,stripped + Q_loss

### Phases

- **Phase 1**: First-Principles Rate-Based Column Equations
- **Phase 2**: Synthetic Data Generation & Dynamic Simulation  
- **Phase 3**: LSTM Soft Sensor Training & Validation
- **Phase 4**: Streamlit Control & Monitoring Dashboard

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
streamlit run dashboard.py
```