"""Bidirectional recurrent heads used atop the U-Net's flattened conv output.

`BiGRU` is what the shipped checkpoint uses (model.py's `E2E0`/`E2E`,
`n_gru=1`); `BiLSTM` is unused by any code path here but kept for
architecture parity with upstream variants that swap it in.

Reads: nothing (leaf module).
"""

import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]

