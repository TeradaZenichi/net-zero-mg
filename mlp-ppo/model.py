import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, action_size):
        super(PolicyNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Definir as camadas ocultas
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Definir a camada de saída
        self.output_layer = nn.Linear(hidden_sizes[-1], action_size)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        action = self.output_layer(x)
        action = F.relu(action)  # Para garantir não negatividade
        return action

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(ValueNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Definir as camadas ocultas
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Definir a camada de saída
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        value = self.output_layer(x)
        return value
