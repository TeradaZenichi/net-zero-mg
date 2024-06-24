import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepMLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Definir a primeira camada oculta
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Definir as camadas ocultas subsequentes
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Definir a camada de saída
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    # Exemplo de uso do modelo
    N = 5  # Número de amostras anteriores de carga e PV
    input_size = 2 * N + 1  # 2*N para carga e PV, 1 para SoC atual
    hidden_sizes = [128, 64]  # Tamanho das camadas ocultas
    output_size = 3  # Saídas: bess_power, pv_shedding, load_shedding

    model = DeepMLP(input_size, hidden_sizes, output_size)
    print(model)

    # Exemplo de entrada aleatória para o modelo
    example_input = torch.randn(1, input_size)
    example_output = model(example_input)
    print(example_output)
