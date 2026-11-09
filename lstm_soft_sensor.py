"""
LSTM Soft Sensor for MEA Absorption Column
Predicts CO2 capture efficiency and overhead purity from sparse temperature sensor data.
"""

import torch
import torch.nn as nn


class MEASoftSensorLSTM(nn.Module):
    """
    LSTM-based soft sensor for MEA absorption column.
    
    Inputs (X_t): Time-series data from 5 sparse temperature sensors across the column trays
                  (T_1, T_3, T_5, T_7, T_10), rich amine feed flow rate, and flue gas inlet temperature.
    
    Outputs (Y_t): Instantaneous CO2 capture efficiency (%) and clean overhead purity/slippage.
    """
    
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.2):
        super(MEASoftSensorLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the LSTM soft sensor.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tensor of shape (batch_size, output_dim) - predicted CO2 capture efficiency and purity
        """
        # out shape: (batch_size, seq_len, hidden_dim)
        out, (hn, cn) = self.lstm(x)
        
        # Take the last time-step output
        # out[:, -1, :] shape: (batch_size, hidden_dim)
        out = self.dropout(out[:, -1, :])
        
        # Final fully connected layer
        out = self.fc(out)  # shape: (batch_size, output_dim)
        
        return out


def create_model(input_dim=7, hidden_dim=64, output_dim=2, num_layers=2, dropout=0.2):
    """Factory function to create the LSTM soft sensor model."""
    model = MEASoftSensorLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    model = create_model()
    print(f"Model created: {model}")
    
    # Test forward pass with dummy data
    # batch_size=4, seq_len=10, input_dim=7
    x = torch.randn(4, 10, 7)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # Should be (4, 2)
    print(f"Output values: {y}")