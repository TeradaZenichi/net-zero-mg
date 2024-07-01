import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from model_utils import QNetwork, ReplayBuffer, save_model

# Load the CSV files
file_path_pv = 'MVS/df_unicamp_train.csv'
pv_data = pd.read_csv(file_path_pv, delimiter=';')
pv_data_sets = pv_data.drop(columns='PerfilPV').values.tolist()

file_path_demand = 'MVS/household_power_train.csv'
demand_data = pd.read_csv(file_path_demand, delimiter=';')
demand_data_sets = demand_data.drop(columns='PerfilD').values.tolist()

max_soc = 80
min_soc = 10
max_power = 20
min_power = -20
deltat = 24/288

class BatteryEnv:
    def __init__(self, pv_data, demand_data, initial_soc, max_soc=80, min_soc=10):
        self.pv_data = pv_data
        self.demand_data = demand_data
        self.initial_soc = initial_soc
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.soc = initial_soc
        self.state = (0, self.soc)
        self.time = 0
        self.max_time = len(pv_data)

        self.load_shedding_cost = self.create_cost_array([8, 8, 8])
        self.injection_cost = self.create_cost_array([5, 3, 6])
        self.substation_supply_cost = self.create_cost_array([5, 3, 6])

    def create_cost_array(self, costs):
        third = self.max_time // 3
        cost_array = np.zeros(self.max_time)
        cost_array[:third] = costs[0]
        cost_array[third:2*third] = costs[1]
        cost_array[2*third:] = costs[2]
        return cost_array

    def reset(self):
        self.soc = self.initial_soc
        self.time = 0
        self.state = (self.pv_data[self.time] - self.demand_data[self.time], self.soc)
        return self.state

    def step(self, action):
        pv_demand_diff = self.pv_data[self.time] - self.demand_data[self.time]
        reward = 0
        load_shedding = 0
        injection = 0
        substation_supply = 0
        charge_power = 0
        discharge_power = 0

        load_shedding_cost = self.load_shedding_cost[self.time]
        injection_cost = self.injection_cost[self.time]
        substation_supply_cost = self.substation_supply_cost[self.time]

        # Map action to power value
        power_value = min_power + action * (max_power - min_power) / 1000

        if power_value > 0:  # Charging
            if self.pv_data[self.time] > self.demand_data[self.time]:
                new_soc = self.soc + deltat * power_value * 0.95
                if new_soc > self.max_soc:
                    power_value2 = (self.max_soc - self.soc) / (0.95 * deltat)
                    if power_value2 < pv_demand_diff:
                        charge_power = power_value2
                        injection =  pv_demand_diff - charge_power
                        self.soc = self.max_soc
                        reward = -injection_cost * injection + (charge_power - power_value)
                    else:
                        charge_power = pv_demand_diff
                        injection = 0
                        self.soc = self.soc + deltat * charge_power * 0.95
                        reward = (charge_power - power_value)                        
                else:
                    if power_value < pv_demand_diff:
                        charge_power = power_value
                        injection =  pv_demand_diff - charge_power
                        self.soc = new_soc
                        reward = -injection_cost * injection  
                    else:
                        charge_power = pv_demand_diff
                        injection = 0
                        self.soc = self.soc + deltat * charge_power * 0.95
                        reward = (charge_power - power_value) 
            else:
                charge_power = 0
                discharge_power = 0
                if load_shedding_cost <= substation_supply_cost:
                    load_shedding = self.demand_data[self.time] - self.pv_data[self.time]
                    reward = -load_shedding_cost * load_shedding  - power_value 
                else:
                    substation_supply = self.demand_data[self.time] - self.pv_data[self.time]
                    reward = -substation_supply_cost * substation_supply  - power_value 

        elif power_value < 0:  # Discharging
            power_value = abs(power_value)
            if self.pv_data[self.time] < self.demand_data[self.time]:
                new_soc = self.soc - deltat * power_value / 0.95
                if new_soc >= self.min_soc:
                    if power_value <= - pv_demand_diff:
                        discharge_power = power_value
                        self.soc = new_soc
                        substation_supply = - pv_demand_diff - power_value
                        reward = - substation_supply_cost * substation_supply
                    else:
                        discharge_power =  - pv_demand_diff
                        self.soc = self.soc - deltat * discharge_power / 0.95
                        reward = (discharge_power - power_value) 
                else:
                    power_value2 = (self.soc - self.min_soc) * 0.95 / deltat
                    if power_value2 <= - pv_demand_diff:
                        discharge_power = power_value2
                        self.soc = self.min_soc
                        substation_supply = - pv_demand_diff - discharge_power
                        reward = - substation_supply_cost * substation_supply + (discharge_power - power_value) 
                    else:
                        discharge_power =  - pv_demand_diff
                        self.soc = self.soc - deltat * discharge_power / 0.95
                        reward = (discharge_power - power_value) 
            else:
                injection = self.pv_data[self.time] - self.demand_data[self.time]
                reward = -injection_cost * injection - power_value 

        else:  # Idle
            if self.pv_data[self.time] > self.demand_data[self.time]:
                injection = self.pv_data[self.time] - self.demand_data[self.time]
                reward = -injection_cost * injection
            else:
                if load_shedding_cost <= substation_supply_cost:
                    load_shedding = self.demand_data[self.time] - self.pv_data[self.time]
                    reward = -load_shedding_cost * load_shedding
                else:
                    substation_supply = self.demand_data[self.time] - self.pv_data[self.time]
                    reward = -substation_supply_cost * substation_supply

        self.time += 1
        if self.time >= self.max_time:
            done = True
        else:
            done = False
            self.state = (self.pv_data[self.time] - self.demand_data[self.time], self.soc)

        return self.state, reward, done, load_shedding, injection, substation_supply, charge_power, discharge_power

state_dim = 2
action_dim = 1001  # 500 for charging, 500 for discharging, 1 for idle
batch_size = 64
lr = 0.001
gamma = 0.75
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
num_episodes = 1000
target_update = 10
buffer_size = 10000

policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
replay_buffer = ReplayBuffer(buffer_size)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(range(1001))
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()

episode_rewards = []

for episode in range(num_episodes):
    # Select a random training set for each episode
    data_index = random.randint(0, len(pv_data_sets) - 1)
    initial_soc = random.randint(10, 30)
    env = BatteryEnv(pv_data_sets[data_index], demand_data_sets[data_index], initial_soc, max_soc, min_soc)

    state = env.reset()
    state = np.array([(state[0] + 24) / 48.0, state[1] / 80.0])
    done = False
    episode_reward = 0
    epsilon = max(epsilon_end, epsilon_decay * epsilon_start)

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, load_shedding, injection, substation_supply, charge_power, discharge_power = env.step(action)
        next_state = np.array([(next_state[0] + 24) / 48.0, next_state[1] / 80.0])

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0]
            targets = rewards + (gamma * next_q_values * (1 - dones))

            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    episode_rewards.append(episode_reward)
    epsilon_start = epsilon

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode + 1}: Total Reward: {episode_reward:.3f}")

plt.figure(figsize=(12, 6))
plt.plot(range(1, num_episodes + 1), episode_rewards, label='Episode Reward', color='orange')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.show()

# Save the model
save_model(policy_net, 'battery_policy_net.pth')
