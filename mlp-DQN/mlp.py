import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, PBESSmax):
        super(DeepMLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Definir a primeira camada oculta
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Definir as camadas ocultas subsequentes
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Definir a camada de saída
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        
        # Definir o valor máximo de potência do BESS
        self.PBESSmax = PBESSmax
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        x = F.relu(x)  # Aplicar ReLU para garantir não negatividade
        x = torch.clamp(x, max=2*self.PBESSmax)  # Garantir que a saída não exceda PBESSmax
        return x

if __name__ == "__main__":
    # Exemplo de uso do modelo
    N = 5  # Número de amostras anteriores de carga e PV
    input_size = 2 * N + 1  # 2*N para carga e PV, 1 para SoC atual
    hidden_sizes = [128, 64]  # Tamanho das camadas ocultas
    PBESSmax = 10  # Valor máximo de potência do BESS

    model = DeepMLP(input_size, hidden_sizes, PBESSmax)
    print(model)

    # Exemplo de entrada aleatória para o modelo
    example_input = torch.randn(1, input_size)
    example_output = model(example_input)
    print(example_output)
