import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from env import NetZeroMicrogridEnv
from mlp import DeepMLP
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
    pv_series, load_series, timestamps = load_test_data(pv_file, load_file, T, time_step_minutes)

    # Definir o modelo MLP
    N = config["N"]  # Número de amostras anteriores de carga e PV
    input_size = 2 * N + 1  # 2*N para carga e PV, 1 para SoC atual
    hidden_sizes = [128, 64]  # Tamanho das camadas ocultas
    PBESSmax = env.BESSPmax  # Valor máximo de potência do BESS

    model = DeepMLP(input_size, hidden_sizes, PBESSmax)

    # Carregar o modelo treinado
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()  # Modo de avaliação

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

        prev_loads = prev_loads * env.demand_capacity
        prev_pv = prev_pv * env.pv_capacity
        
        # Preparar a entrada do modelo
        model_input = np.concatenate((prev_loads, prev_pv, [state[0]]))
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

        # Armazenar os resultados
        results.append({
            'timestamp': timestamps[step],
            'PEDS': next_state[1],
            'PV': pv_power,
            'Load': load_power,
            'PV_Shedding': -pv_power * (1 - next_state[4]),  # Ajustado para refletir pv_shedding
            'Load_Shedding': load_power * (1 - next_state[6]),  # Ajustado para refletir load_shedding
            'BESS_Power': bess_power - env.BESSPmax,
            'BESS_Power_real': next_state[10],
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

    # Remover o arquivo temporário de configuração
    os.remove(temp_config_path)
