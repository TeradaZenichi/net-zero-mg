import torch
import torch.nn as nn
import torch.optim as optim
from env import NetZeroMicrogridEnv
from mlp import DeepMLP
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

# Função para ler os arquivos CSV e alinhar os timestamps
def load_data(pv_file, load_file, T, time_step_minutes):
    # Carregar dados de geração PV
    pv_data = pd.read_csv(pv_file, sep=';', parse_dates=['datetime'], index_col='datetime')
    
    # Carregar dados de demanda
    load_data = pd.read_csv(load_file, sep=';', parse_dates=['datetime'], index_col='datetime')

    # Verificar se os timestamps estão alinhados
    if not pv_data.index.equals(load_data.index):
        raise ValueError("Os timestamps dos arquivos PV e Load não estão alinhados.")
    
    # Calcular o número de intervalos no timestep dado em T meses
    num_intervals_per_hour = 60 // time_step_minutes
    num_intervals_per_day = 24 * num_intervals_per_hour
    num_intervals_per_month = 30 * num_intervals_per_day
    num_intervals = T * num_intervals_per_month

    # Selecionar os primeiros T meses de dados
    pv_data = pv_data.head(num_intervals)
    load_data = load_data.head(num_intervals)
    
    return pv_data['power'], load_data['Global_active_power']

# Função para obter amostras anteriores
def get_previous_samples(series, step, N):
    if step < N:
        return np.zeros(N)
    else:
        return series.iloc[step-N:step].values

if __name__ == "__main__":
    # Carregar parâmetros do arquivo config.json
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Salvar os parâmetros em um arquivo temporário
    temp_config_path = 'temp_config.json'
    with open(temp_config_path, 'w') as f:
        json.dump(config, f)

    # Caminhos dos arquivos CSV
    pv_file = 'data/df_unicamp_5min.csv'
    load_file = 'data/household_power_consumption_5min.csv'
    T = config["T"]  # Número de meses de dados a serem usados

    # Inicializar o ambiente
    env = NetZeroMicrogridEnv(temp_config_path)
    time_step_minutes = config["time_step"]  # Obter o timestep do ambiente

    # Carregar dados
    pv_series, load_series = load_data(pv_file, load_file, T, time_step_minutes)

    # Definir o modelo MLP
    N = config["N"]  # Número de amostras anteriores de carga e PV
    input_size = 2 * N + 1  # 2*N para carga e PV, 1 para SoC atual
    hidden_sizes = [128, 64]  # Tamanho das camadas ocultas
    PBESSmax = env.BESSPmax  # Valor máximo de potência do BESS

    model = DeepMLP(input_size, hidden_sizes, 1)  # Atualizado para ter apenas 1 saída
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_episodes = 1000
    gamma = 0.99

    print("Iniciando o treinamento...")
    
    for episode in tqdm(range(num_episodes), desc="Episódios"):
        state = env.reset()
        done = False
        total_reward = 0
        step = N  # Garantir que haja dados suficientes para amostras anteriores
        max_steps = len(pv_series)  # Número máximo de passos baseado na quantidade de dados carregados

        while not done and step < max_steps:
            # Obter os valores de PV e Load a partir das séries temporais
            pv_power_normalized = pv_series.iloc[step]
            load_power_normalized = load_series.iloc[step]

            # Desnormalizar os valores de PV e Load
            pv_power = pv_power_normalized * env.pv_capacity
            load_power = load_power_normalized * env.demand_capacity

            # Obter amostras anteriores
            prev_loads = get_previous_samples(load_series, step, N)
            prev_pv = get_previous_samples(pv_series, step, N)

            prev_loads = prev_loads * env.demand_capacity
            prev_pv = prev_pv * env.pv_capacity

            # Preparar a entrada do modelo
            model_input = np.concatenate((prev_loads, prev_pv, [state[0]])).astype(np.float32)
            model_input = torch.FloatTensor(model_input).unsqueeze(0)

            # Obter a ação do modelo
            model_output = model(model_input)
            bess_power = model_output.item()

            # Verificação para NaN
            if np.isnan(bess_power):
                bess_power = 0.0

            # Garantir que as ações estejam dentro dos limites
            bess_power = np.clip(bess_power, 0, 2 * env.BESSPmax)

            action = (bess_power, 0, 0)  # pv_shedding e load_shedding definidos como 0

            # Executar a ação no ambiente
            next_state, reward, done, info = env.step(action, pv_power, load_power)
            total_reward += reward

            # Preparar a entrada do modelo para o próximo estado
            next_model_input = np.concatenate((get_previous_samples(load_series, step+1, N), get_previous_samples(pv_series, step+1, N), [next_state[0]])).astype(np.float32)
            next_model_input = torch.FloatTensor(next_model_input).unsqueeze(0)

            # Calcular o valor alvo
            target = reward + gamma * model(next_model_input).item() * (not done)
            target = torch.FloatTensor([target])

            # Prever o valor atual
            prediction = model(model_input)

            # Calcular a perda
            loss = criterion(prediction, target)

            # Atualizar o modelo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Atualizar o estado
            state = next_state

            # Incrementar o passo
            step += 1
            if step % 1000 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


        
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    print("Treinamento concluído.")
    # Salvar o modelo treinado
    torch.save(model.state_dict(), 'trained_model.pth')

    # Remover o arquivo temporário de configuração
    os.remove(temp_config_path)
