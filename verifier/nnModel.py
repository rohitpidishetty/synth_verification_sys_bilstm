import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        scores = self.attention(lstm_output) 
        weights = F.softmax(scores, dim=1)   
        context = torch.sum(weights * lstm_output, dim=1) 
        return context, weights

class BiLSTM_Attention_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, num_layers=1, dropout=0.3):
        super(BiLSTM_Attention_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_dim * 2)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) 
        lstm_out, _ = self.lstm(x)  
        context, attn_weights = self.attention(lstm_out)
        context = self.layer_norm(context)
        context = self.dropout(context)
        out = self.fc(context)
        return out, attn_weights
