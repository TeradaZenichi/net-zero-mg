import pandas as pd

# Load the CSV file with the correct delimiter
file_path = 'MVS/df_unicamp_train.csv'
data = pd.read_csv(file_path, delimiter=';')

# Extract the data excluding the 'PerfilPV' column and convert to a list of lists
pv_data_sets = data.drop(columns='PerfilPV').values.tolist()

# Display the first few sets to verify the transformation
for i, pv_set in enumerate(pv_data_sets[:4], start=1):
    print(f"PV Data Set {i}: {pv_set}")

# Load the CSV file with the correct delimiter
file_path = 'MVS/household_power_train.csv'
data = pd.read_csv(file_path, delimiter=';')

# Extract the data excluding the 'PerfilD' column and convert to a list of lists
demand_data_sets = data.drop(columns='PerfilD').values.tolist()

# Display the first few sets to verify the transformation
for i, demand_set in enumerate(demand_data_sets[:4], start=1):
    print(f"Demand Data Set {i}: {demand_set}")

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

max_soc = 80
min_soc = 10
max_power = 20
min_power = -20

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
        self.injection_cost = self.create_cost_array([3, 5, 6])
        self.substation_supply_cost = self.create_cost_array([3, 5, 6])

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
                    charge_power = (self.max_soc - self.soc) / (0.95 * deltat)
                    injection =  pv_demand_diff - charge_power
                    self.soc = self.max_soc
                    reward = -injection_cost * injection
                else:
                    if pv_demand_diff - power_value >=0:
                        charge_power = power_value
                        injection =  pv_demand_diff - charge_power
                        reward = -injection_cost * injection
                        self.soc = new_soc
                    else:
                        charge_power = pv_demand_diff
                        injection = 0
                        reward = 1
                        self.soc = self.soc + deltat * charge_power * 0.95
            else:
                charge_power = 0
                discharge_power = 0
                if load_shedding_cost <= substation_supply_cost:
                    load_shedding = self.demand_data[self.time] - self.pv_data[self.time]
                    reward = -load_shedding_cost * load_shedding
                else:
                    substation_supply = self.demand_data[self.time] - self.pv_data[self.time]
                    reward = -substation_supply_cost * substation_supply

        elif power_value < 0:  # Discharging
            power_value = abs(power_value)
            if self.pv_data[self.time] < self.demand_data[self.time]:
                new_soc = self.soc - deltat * power_value / 0.95
                if new_soc >= self.min_soc:
                    if power_value <= - pv_demand_diff:
                        discharge_power = power_value
                        self.soc = new_soc
                        substation_supply = - pv_demand_diff - power_value
                        reward = -substation_supply_cost * substation_supply
                    else:
                        discharge_power =  - pv_demand_diff
                        self.soc = self.soc - deltat * discharge_power / 0.95
                        reward = 1
                else:
                    power_value = (self.soc - self.min_soc) * 0.95 / deltat
                    if power_value <= - pv_demand_diff:
                        discharge_power = power_value
                        self.soc = self.min_soc
                        substation_supply = - pv_demand_diff - discharge_power
                        reward = -substation_supply_cost * substation_supply
                    else:
                        discharge_power =  - pv_demand_diff
                        self.soc = self.soc - deltat * discharge_power / 0.95
                        reward = 1

                    # if load_shedding_cost <= substation_supply_cost:
                    #     load_shedding = - pv_demand_diff - discharge_power
                    #     reward = -load_shedding_cost * load_shedding
                    # else:
                    #     substation_supply = - pv_demand_diff - discharge_power
                    #     reward = -substation_supply_cost * substation_supply
                    # self.soc = self.min_soc
            else:
                injection = self.pv_data[self.time] - self.demand_data[self.time]
                reward = -injection_cost * injection

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

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience',
                                     field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)

        states = torch.tensor([e.state for e in experiences], dtype=torch.float32)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.tensor([e.next_state for e in experiences], dtype=torch.float32)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

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
deltat = 24/288

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
    data_index = random.randint(0, 1006)
    initial_soc = random.randint(10, 100)
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

# Test on the fourth dataset with a random initial SOC between 10 and 30
initial_soc = random.randint(10, 30)
env = BatteryEnv(pv_data_sets[1007], demand_data_sets[1007], initial_soc, max_soc, min_soc)
state = env.reset()
state = np.array([(state[0] + 24) / 48.0, state[1] / 80.0])
done = False
total_reward = 0

soc_values = []
load_shedding_values = []
injection_values = []
substation_supply_values = []
charge_power_values = []
discharge_power_values = []
actions = []
demand_values = []
pv_values = []

print(f"\nTesting after {num_episodes} episodes with initial SOC: {initial_soc}")
print(f"{'Time':<5}{'PV':<8}{'Demand':<8}{'SOC':<8}{'Action':<8}{'Decision':<20}{'Reward':<10}{'Load Shedding':<15}{'Injection':<10}{'Substation Supply':<18}")

while not done:
    action = select_action(state, epsilon_end)
    next_state, reward, done, load_shedding, injection, substation_supply, charge_power, discharge_power = env.step(action)
    next_state = np.array([(next_state[0] + 24) / 48.0, next_state[1] / 80.0])
    total_reward += reward

    if action == 500:
        decision = "Idle"
    elif action < 500:
        decision = f"Discharge {min_power + action * (max_power - min_power) / 1000:.2f}"
    else:
        decision = f"Charge {min_power + action * (max_power - min_power) / 1000:.2f}"

    soc_values.append(next_state[1] * 80.0)
    load_shedding_values.append(load_shedding)
    injection_values.append(injection)
    substation_supply_values.append(substation_supply)
    charge_power_values.append(charge_power)
    discharge_power_values.append(discharge_power)
    actions.append(action)
    demand_values.append(env.demand_data[env.time-1])
    pv_values.append(env.pv_data[env.time-1])

    print(f"{env.time-1:<5}{env.pv_data[env.time-1]:<8.2f}{env.demand_data[env.time-1]:<8.2f}{state[1] * 80.0:<8.2f}{action:<8}{decision:<20}{reward:<10.2f}{load_shedding:<15.2f}{injection:<10.2f}{substation_supply:<18.2f}")

    state = next_state

print(f"Total reward after {num_episodes} episodes: {total_reward:.2f}\n")

time_steps = np.arange(1, env.max_time + 1)
width = 0.40

plt.figure(figsize=(12, 6), dpi=200)

plt.bar(time_steps, pv_values, color='orange', width=width, label='PV')
plt.bar(time_steps, substation_supply_values, color='green', width=width, bottom=pv_values, label='PS')
plt.bar(time_steps, discharge_power_values, color='purple', width=width, bottom=np.array(pv_values) + np.array(substation_supply_values), label='Pdc')
plt.bar(time_steps, np.array(charge_power_values) * -1, color='red', width=width, label='Pch')
plt.bar(time_steps, np.array(injection_values) * -1, color='blue', bottom=np.array(charge_power_values) * -1, width=width, label='PEx')
plt.plot(time_steps, demand_values, color='black', label='D', linewidth=0.5)
# plt.plot(time_steps, np.array(demand_values) - np.array(load_shedding_values), marker='*', color='brown', label='LS', linewidth=2)

plt.xlabel('Time Step')
plt.ylim(-20, 35)
plt.ylabel('Value')
plt.title('Battery Management Over Time')
plt.xticks(time_steps)
plt.legend(loc='best', ncol=7)
plt.xticks(time_steps[::10])  # Adjust the tick frequency here
plt.grid(False)
plt.show()

plt.figure(figsize=(12, 6), dpi=200)
plt.plot(time_steps, soc_values, color='blue', label='SOC')
plt.xlabel('Time Step')
plt.ylabel('State of Charge (SOC)')
plt.title('SOC Behavior Over Time')
plt.xticks(time_steps)
plt.legend(loc='best')
plt.xticks(time_steps[::10])  # Adjust the tick frequency here
plt.grid(False)
plt.show()
