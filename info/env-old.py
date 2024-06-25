import gym
from gym import spaces
import numpy as np
import json

class NetZeroMicrogridEnv(gym.Env):
    def __init__(self, config_path='config.json'):
        super(NetZeroMicrogridEnv, self).__init__()
        
        # Carregando parâmetros do arquivo JSON
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        self.grid_cost_import = config['grid_cost_import']
        self.grid_cost_export = config['grid_cost_export']
        self.load_shedding_penalty = config['load_shedding_penalty']
        self.initial_soc = config['initial_soc']
        self.soc_min = config['soc_min']
        self.soc_max = config['soc_max']
        self.off_grid_probability = config['off_grid_probability']
        self.eta_ac_dc = config['eta_ac_dc']
        self.eta_dc_ac = config['eta_dc_ac']
        self.soc_violation_penalty = config['soc_violation_penalty']
        self.max_import_power = config['max_import_power']
        self.import_violation_penalty = config['import_violation_penalty']
        self.pv_capacity = config['pv_capacity']
        self.demand_capacity = config['demand_capacity']
        self.time_step_minutes = config['time_step']  # time_step em minutos
        self.off_grid_duration_hours = config['off_grid_duration_hours']
        self.off_grid_duration_steps = int(self.off_grid_duration_hours * 60 / self.time_step_minutes)
        self.BESSPmax = config['BESSPmax']
        self.BESSEmax = config['BESSEmax']  # Adicionado para considerar a energia máxima do BESS
        
        self.current_step = 0
        self.off_grid_counter = 0
        
        # Definindo o espaço de ação
        self.action_space = spaces.Tuple((
            spaces.Box(low=-self.BESSPmax, high=self.BESSPmax, shape=(1,), dtype=np.float32),  # Despacho da bateria em passos decimais de -BESSPmax a BESSPmax
            spaces.Discrete(2),  # Corte do PV
            spaces.Discrete(11)  # Corte da carga (0 a 100% em passos de 10%)
        ))
        
        # Definindo o espaço de observação
        self.observation_space = spaces.Box(
            low=np.array([self.soc_min, 0, 0, -self.max_import_power, 0, 0, 0, 0, self.soc_min, 0]),
            high=np.array([self.soc_max, self.pv_capacity, self.demand_capacity, self.max_import_power, self.pv_capacity, self.demand_capacity, self.demand_capacity, np.inf, self.soc_max, 1]),
            dtype=np.float32
        )
        
        # Estado inicial
        self.state = self.reset()
        
    def reset(self):
        self.current_step = 0
        self.off_grid_counter = 0
        self.state = np.array([self.initial_soc * self.BESSEmax, 0, 0, 0, 0, 0, 0, 0, self.initial_soc, 0])
        return self.state
    
    def _apply_noise(self, power):
        noise = np.random.uniform(0, 0.05) * abs(power)
        return power - noise
    
    def _max_charge_power(self, soc):
        if soc < 0.2:
            return self.BESSPmax
        elif soc < 0.8:
            return self.BESSPmax * (1 - (soc - 0.2) / 0.6)
        else:
            return self.BESSPmax * 0.2
    
    def step(self, action, pv_power, load_power):
        bess_power = action[0]
        load_shedding = action[1]
        pv_shedding = action[2]
        
        # Verificar se estamos em estado off-grid
        if self.off_grid_counter > 0:
            off_grid = True
            self.off_grid_counter -= 1
        else:
            off_grid = np.random.rand() < self.off_grid_probability
            if off_grid:
                self.off_grid_counter = self.off_grid_duration_steps - 1
        
        # Aplicando ruído à ação da bateria
        bess_power = self._apply_noise(bess_power)
        
        # Verificando se a potência da bateria excede o valor máximo permitido
        power_violation_penalty = 0
        if abs(bess_power) > self.BESSPmax:
            power_violation_penalty = self.import_violation_penalty
            bess_power = np.sign(bess_power) * self.BESSPmax
        
        # Calculando a potência máxima de carregamento baseada no SoC
        max_charge_power = self._max_charge_power(self.state[0] / self.BESSEmax)
        bess_power = min(bess_power, max_charge_power)
        
        # Atualizando o SOC da bateria
        soc_change = bess_power * (self.time_step_minutes / 60)  # Convertendo time_step de minutos para horas
        new_soc = self.state[0] + soc_change

        # Calculando o balanço de energia
        net_energy = self.eta_ac_dc * (bess_power if bess_power > 0 else 0) + pv_power * (1 - pv_shedding) - \
                     self.eta_dc_ac * (bess_power if bess_power < 0 else 0) - load_power * (1 - load_shedding)
        
        # Calculando a potência da rede
        if off_grid:
            grid_power = 0
        else:
            grid_power = -net_energy
        
        # Calculando custos e recompensas
        if off_grid:
            # Caso off-grid, não pode importar ou exportar energia
            grid_cost = 0
            if net_energy < 0:
                load_shedding_cost = abs(net_energy) * self.load_shedding_penalty
            else:
                load_shedding_cost = 0
        else:
            if grid_power > 0:
                grid_cost = grid_power * self.grid_cost_import
            else:
                grid_cost = -grid_power * self.grid_cost_export
            load_shedding_cost = load_shedding * self.load_shedding_penalty
        
        # Aplicando penalidades por violação do SoC
        soc_violation_penalty = 0
        if new_soc < self.soc_min * self.BESSEmax:
            soc_violation_penalty = self.soc_violation_penalty * (self.soc_min * self.BESSEmax - new_soc)
        elif new_soc > self.soc_max * self.BESSEmax:
            soc_violation_penalty = self.soc_violation_penalty * (new_soc - self.soc_max * self.BESSEmax)
        
        reward = - (grid_cost + load_shedding_cost + power_violation_penalty + soc_violation_penalty)
        
        # Atualizando o estado
        self.state = np.array([
            new_soc,                         # Posição 0: Novo SoC da bateria
            pv_power,                        # Posição 1: Potência PV (instantâneo)
            load_power,                      # Posição 2: Potência da Carga (instantâneo)
            grid_power,                      # Posição 3: Potência da Rede
            pv_power * (1 - pv_shedding),    # Posição 4: PV total após corte
            load_power,                      # Posição 5: Potência da Carga (instantâneo)
            load_power * load_shedding,      # Posição 6: Load shedding
            grid_cost + load_shedding_cost,  # Posição 7: Custo com base no time_step
            new_soc / self.BESSEmax,         # Posição 8: SoC do BESS (normalizado)
            1 if off_grid else 0             # Posição 9: Estado da Rede (1 se off-grid, 0 se on-grid)
        ])
        
        # Verificação de término do episódio
        done = (new_soc < self.soc_min * self.BESSEmax or new_soc > self.soc_max * self.BESSEmax)
        done_reason = ""
        if new_soc < self.soc_min * self.BESSEmax:
            done_reason = "SOC below minimum threshold"
        elif new_soc > self.soc_max * self.BESSEmax:
            done_reason = "SOC above maximum threshold"
        
        # Informação adicional
        info = {
            'grid_cost': grid_cost,
            'load_shedding_cost': load_shedding_cost,
            'off_grid': off_grid,
            'import_power': grid_power if grid_power > 0 else 0,
            'export_power': -grid_power if grid_power < 0 else 0,
            'done_reason': done_reason if done else ""
        }
        
        return self.state, reward, done, info

    def render(self, mode='human'):
        print(f"State: {self.state}")

# Para testar o ambiente, remova ou comente a linha abaixo quando importar no main.py
# if __name__ == "__main__":
#     env = NetZeroMicrogridEnv()
# 
#     state = env.reset()
#     done = False
#     total_reward = 0
#     
#     while not done:
#         pv_power = np.random.uniform(0, env.pv_capacity)  # Exemplo de geração PV aleatória
#         load_power = np.random.uniform(0, env.demand_capacity)  # Exemplo de demanda aleatória
#         action = env.action_space.sample()  # Ação aleatória
#         next_state, reward, done, info = env.step(action, pv_power, load_power)
#         total_reward += reward
#         env.render()
#     
#     print(f"Total Reward: {total_reward}")
#     print(f"State History: {env.state}")
#     print(f"Info: {info}")
