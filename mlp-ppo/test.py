import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from env import NetZeroMicrogridEnv
from model import PolicyNetwork
import matplotlib.pyplot as plt
import json
import os

# Função para ler os arquivos CSV e alinhar os timestamps
def load_test_data(pv_file, load_file, T, time_step_minutes):
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

    # Selecionar os últimos T meses de dados
    pv_data = pv_data.tail(num_intervals)
    load_data = load_data.tail(num_intervals)
    
    return pv_data['power'], load_data['Global_active_power'], pv_data.index

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

    # Caminhos dos arquivos CSV
    pv_file = 'data/df_unicamp_5min.csv'
    load_file = 'data/household_power_consumption_5min.csv'
    T = config["T"]  # Número de meses de dados a serem usados

    # Inicializar o ambiente
    env = NetZeroMicrogridEnv('config.json')
    time_step_minutes = config["time_step"]  # Obter o timestep do ambiente

    # Carregar dados
    pv_series, load_series, timestamps = load_test_data(pv_file, load_file, T, time_step_minutes)

    # Definir o modelo
    N = config["N"]  # Número de amostras anteriores de carga e PV
    input_size = 2 * N + 1  # 2*N para carga e PV, 1 para SoC atual
    hidden_sizes = [128, 64]
    action_size = 1

    policy_net = PolicyNetwork(input_size, hidden_sizes, action_size)

    # Carregar o modelo treinado
    policy_net.load_state_dict(torch.load('trained_policy_net.pth'))
    policy_net.eval()  # Modo de avaliação

    total_reward = 0
    step = N  # Garantir que haja dados suficientes para amostras anteriores
    max_steps = len(pv_series)  # Número máximo de passos baseado na quantidade de dados carregados

    state = env.reset()

    # DataFrame para armazenar os resultados
    results = []

    while step < max_steps:
        # Obter os valores de PV e Load a partir das séries temporais
        pv_power_normalized = pv_series.iloc[step]
        load_power_normalized = load_series.iloc[step]

        # Desnormalizar os valores de PV e Load
        pv_power = pv_power_normalized * env.pv_capacity
        load_power = load_power_normalized * env.demand_capacity

        # Obter amostras anteriores
        prev_loads = get_previous_samples(load_series, step, N)
        prev_pv = get_previous_samples(pv_series, step, N)

        # Preparar a entrada do modelo
        model_input = np.concatenate((prev_loads, prev_pv, [state[0]]))
        model_input = torch.FloatTensor(model_input).unsqueeze(0)

        # Obter a ação do modelo
        action_probs = policy_net(model_input)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()

        bess_power = action.item() * (2 * env.BESSPmax / action_size) - env.BESSPmax

        # Executar a ação no ambiente
        next_state, reward, done, info = env.step([bess_power], pv_power, load_power)
        total_reward += reward

        # Armazenar os resultados
        results.append({
            'timestamp': timestamps[step],
            'PEDS': next_state[1],
            'PV_Shedding': -pv_power * (1 - info.get('pv_shedding', 0)),
            'Load_Shedding': load_power * info.get('load_shedding', 0),
            'BESS_Power': bess_power,
            'SoC': next_state[0]
        })

        # Atualizar o estado
        state = next_state

        # Incrementar o passo
        step += 1

    print(f"Total Reward for Test Data: {total_reward}")

    # Converter os resultados para um DataFrame
    df_results = pd.DataFrame(results)
    df_results.set_index('timestamp', inplace=True)
    df_results.to_csv('test_results.csv')

    # Plotar gráficos
    typical_day = df_results.head(24 * 60 // time_step_minutes)  # Primeiro dia típico
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(typical_day.index, typical_day['PEDS'], label='PEDS')
    plt.plot(typical_day.index, typical_day['PV_Shedding'], label='PV Shedding')
    plt.plot(typical_day.index, typical_day['Load_Shedding'], label='Load Shedding')
    plt.plot(typical_day.index, typical_day['BESS_Power'], label='BESS Power')
    plt.legend()
    plt.title('Operação em um Dia Típico')
    plt.xlabel('Hora')
    plt.ylabel('Potência (kW)')

    plt.subplot(2, 1, 2)
    plt.plot(typical_day.index, typical_day['SoC'], label='SoC', color='purple')
    plt.legend()
    plt.title('Estado de Carga (SoC) da Bateria em um Dia Típico')
    plt.xlabel('Hora')
    plt.ylabel('SoC (%)')

    plt.tight_layout()
    plt.show()
