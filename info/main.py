import numpy as np
import pandas as pd
from env import NetZeroMicrogridEnv

# Função para ler os arquivos CSV e alinhar os timestamps
def load_data(pv_file, load_file):
    # Carregar dados de geração PV
    pv_data = pd.read_csv(pv_file, sep=';', parse_dates=['datetime'], index_col='datetime')
    
    # Carregar dados de demanda
    load_data = pd.read_csv(load_file, sep=';', parse_dates=['datetime'], index_col='datetime')

    # Verificar se os timestamps estão alinhados
    if not pv_data.index.equals(load_data.index):
        raise ValueError("Os timestamps dos arquivos PV e Load não estão alinhados.")
    
    return pv_data['power'], load_data['Global_active_power']

if __name__ == "__main__":
    # Caminhos dos arquivos CSV
    pv_file = 'data/df_unicamp_5min.csv'
    load_file = 'data/household_power_consumption_5min.csv'

    # Carregar dados
    pv_series, load_series = load_data(pv_file, load_file)

    # Inicializar o ambiente
    env = NetZeroMicrogridEnv()

    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Lista para armazenar todos os estados
    all_states = []

    while not done and step < len(pv_series):
        # Obter os valores de PV e Load a partir das séries temporais
        pv_power_normalized = pv_series.iloc[step]
        load_power_normalized = load_series.iloc[step]

        # Desnormalizar os valores de PV e Load
        pv_power = pv_power_normalized * env.pv_capacity
        load_power = load_power_normalized * env.demand_capacity

        # Gerar uma ação aleatória dentro dos limites da bateria
        bess_power = np.random.uniform(-env.BESSPmax, env.BESSPmax)
        pv_shedding = np.random.randint(0, 2)
        load_shedding = np.random.randint(0, 11)

        action = (bess_power, pv_shedding, load_shedding)

        # Executar a ação no ambiente
        next_state, reward, done, info = env.step(action, pv_power, load_power)
        total_reward += reward

        # Renderizar o estado atual (opcional)
        env.render()

        # Armazenar o estado atual
        all_states.append(next_state)

        # Incrementar o passo
        step += 1

    # Converter a lista de estados para um DataFrame
    states_df = pd.DataFrame(all_states, columns=[
        'SoC', 'PV_Power', 'Load_Power', 'Grid_Power', 'PV_Total', 
        'Load', 'Load_Shedding', 'Cost', 'BESS_SoC', 'Grid_State'
    ])

    # Salvar o DataFrame em um arquivo CSV
    states_df.to_csv('all_states.csv', index=False)

    print(f"Total Reward: {total_reward}")
    print(f"State History: {states_df}")
    print(f"Info: {info}")
