import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from model_utils import QNetwork, load_model
from train import BatteryEnv, select_action

# Load the CSV files
file_path_pv = 'MVS/df_unicamp_test.csv'
pv_data = pd.read_csv(file_path_pv, delimiter=';')
pv_data_sets = pv_data.drop(columns='PerfilPV').values.tolist()

file_path_demand = 'MVS/household_power_test.csv'
demand_data = pd.read_csv(file_path_demand, delimiter=';')
demand_data_sets = demand_data.drop(columns='PerfilD').values.tolist()

max_soc = 80
min_soc = 10
max_power = 20
min_power = -20
deltat = 24/288

# Load the trained model
policy_net = QNetwork(2, 1001)
load_model(policy_net, 'battery_policy_net.pth')

# Test on the fourth dataset with a random initial SOC between 10 and 100
initial_soc = random.randint(10, 30)
data_index = random.randint(0, len(pv_data_sets) - 1)
env = BatteryEnv(pv_data_sets[data_index], demand_data_sets[data_index], initial_soc, max_soc, min_soc)
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

print(f"\nTesting after training with initial SOC: {initial_soc}")
print(f"{'Time':<5}{'PV':<8}{'Demand':<8}{'SOC':<8}{'Action':<8}{'Decision':<20}{'Reward':<10}{'Load Shedding':<15}{'Injection':<10}{'Substation Supply':<18}")

while not done:
    action = select_action(state, 0.01)
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

print(f"Total reward after training: {total_reward:.2f}\n")

time_steps = np.arange(1, env.max_time + 1)
width = 0.40

plt.figure(figsize=(12, 6), dpi=200)

plt.bar(time_steps, pv_values, color='orange', width=width, label='PV')
plt.bar(time_steps, substation_supply_values, color='green', width=width, bottom=pv_values, label='PS')
plt.bar(time_steps, discharge_power_values, color='purple', width=width, bottom=np.array(pv_values) + np.array(substation_supply_values), label='Pdc')
plt.bar(time_steps, np.array(charge_power_values) * -1, color='red', width=width, label='Pch')
plt.bar(time_steps, np.array(injection_values) * -1, color='blue', bottom=np.array(charge_power_values) * -1, width=width, label='Pex')
plt.plot(time_steps, demand_values, color='black', label='D', linewidth=0.5)

plt.xlabel('Time Step [min]')
plt.ylim(-20, 35)
plt.ylabel('Value')
plt.title('Energy Management during the day')
plt.xticks(time_steps)
plt.legend(loc='best', ncol=7)
plt.xticks(time_steps[::20])  # Adjust the tick frequency here
plt.grid(False)
plt.show()

plt.figure(figsize=(12, 6), dpi=200)
plt.plot(time_steps, soc_values, color='blue')
plt.xlabel('Time Step [min]')
plt.ylabel('Energy [kWh]')
plt.title('BESS energy over time')
plt.xticks(time_steps)
plt.xticks(time_steps[::20])  # Adjust the tick frequency here
plt.grid(False)
plt.show()
