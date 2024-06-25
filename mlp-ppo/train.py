import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from env import NetZeroMicrogridEnv
from model import PolicyNetwork, ValueNetwork
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    return -torch.min(surr1, surr2).mean()

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

    # Caminhos dos arquivos CSV
    pv_file = 'data/df_unicamp_5min.csv'
    load_file = 'data/household_power_consumption_5min.csv'
    T = config["T"]  # Número de meses de dados a serem usados

    # Inicializar o ambiente
    env = NetZeroMicrogridEnv('config.json')
    time_step_minutes = config["time_step"]  # Obter o timestep do ambiente

    # Carregar dados
    pv_series, load_series = load_data(pv_file, load_file, T, time_step_minutes)

    # Definir o modelo
    N = config["N"]  # Número de amostras anteriores de carga e PV
    input_size = 2 * N + 1  # 2*N para carga e PV, 1 para SoC atual
    hidden_sizes = [128, 64]  # Tamanho das camadas ocultas
    action_size = 1  # Apenas a potência do BESS

    policy_net = PolicyNetwork(input_size, hidden_sizes, action_size)
    value_net = ValueNetwork(input_size, hidden_sizes)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_episodes = 1000
    gamma = 0.99
    epsilon = 0.2
    update_steps = 4
    batch_size = 64

    print("Iniciando o treinamento...")

    for episode in tqdm(range(num_episodes), desc="Episódios"):
        state = env.reset()
        done = False
        total_reward = 0
        step = N  # Garantir que haja dados suficientes para amostras anteriores
        max_steps = len(pv_series)  # Número máximo de passos baseado na quantidade de dados carregados

        states, actions, rewards, log_probs, values = [], [], [], [], []

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

            # Preparar a entrada do modelo
            model_input = np.concatenate((prev_loads, prev_pv, [state[0]]))
            model_input = torch.FloatTensor(model_input).unsqueeze(0)

            # Obter a ação do modelo
            action_probs = policy_net(model_input)

            # Verificação de NaN
            if torch.isnan(action_probs).any():
                raise ValueError(f"action_probs contém NaN: {action_probs}")

            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            value = value_net(model_input)

            bess_power = action.item() * (2 * env.BESSPmax / action_size) - env.BESSPmax

            # Executar a ação no ambiente
            next_state, reward, done, info = env.step([bess_power], pv_power, load_power)
            total_reward += reward

            # Armazenar os dados da trajetória
            states.append(torch.FloatTensor(state))
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)

            # Atualizar o estado
            state = next_state
            step += 1

        # Calcular retornos e vantagens
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        values = torch.cat(values)
        advantages = returns - values.detach()

        # Otimizar a política e o valor
        for _ in range(update_steps):
            for i in range(0, len(states), batch_size):
                batch_indices = slice(i, i + batch_size)
                batch_states = torch.stack(states[batch_indices])
                batch_actions = torch.cat(actions[batch_indices])
                batch_log_probs = torch.cat(log_probs[batch_indices])
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                new_action_probs = policy_net(batch_states)

                # Verificação de NaN
                if torch.isnan(new_action_probs).any():
                    raise ValueError(f"new_action_probs contém NaN: {new_action_probs}")

                new_action_dist = Categorical(new_action_probs)
                new_log_probs = new_action_dist.log_prob(batch_actions)

                policy_loss = ppo_loss(batch_log_probs, new_log_probs, batch_advantages, epsilon)
                value_loss = criterion(value_net(batch_states), batch_returns)

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    print("Treinamento concluído.")
    # Salvar os modelos treinados
    torch.save(policy_net.state_dict(), 'trained_policy_net.pth')
    torch.save(value_net.state_dict(), 'trained_value_net.pth')
