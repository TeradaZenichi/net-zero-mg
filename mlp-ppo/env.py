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
        self.eta_ac_dc = config['eta_ac_dc']  # Eficiência da importação de energia (rede para microrrede)
        self.eta_dc_ac = config['eta_dc_ac']  # Eficiência da exportação de energia (microrrede para rede)
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
        self.action_space = spaces.Box(low=0, high=2 * self.BESSPmax, shape=(1,), dtype=np.float32)  # Despacho da bateria de 0 a 2 * BESSPmax
        
        # Definindo o espaço de observação
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -self.max_import_power, 0, 0, 0, 0, 0, 0]),
            high=np.array([1, self.pv_capacity, self.demand_capacity, self.max_import_power, self.pv_capacity, self.demand_capacity, self.demand_capacity, np.inf, 1, 1]),
            dtype=np.float32
        )
        
        # Estado inicial
        self.state = self.reset()
        
    def reset(self):
        self.current_step = 0
        self.off_grid_counter = 0
        self.state = np.array([self.initial_soc, 0, 0, 0, 0, 0, 0, self.initial_soc, 0], dtype=np.float32)
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
        bess_power = action[0] - self.BESSPmax  # Ajuste para transformar de [0, 2 * BESSPmax] para [-BESSPmax, BESSPmax]
        
        # Verificar se estamos em estado off-grid
        if self.off_grid_counter > 0:
            off_grid = True
            self.off_grid_counter -= 1
        else:
            off_grid = np.random.rand() < self.off_grid_probability
            if off_grid:
                self.off_grid_counter = self.off_grid_duration_steps - 1
        
        # Aplicando ruído à ação da bateria
        # bess_power = self._apply_noise(bess_power)
        
        # Verificando se a potência da bateria excede o valor máximo permitido
        power_violation_penalty = 0
        if abs(bess_power) > self.BESSPmax:
            power_violation_penalty = self.import_violation_penalty
            bess_power = np.sign(bess_power) * self.BESSPmax
        
        # # Calculando a potência máxima de carregamento baseada no SoC
        # max_charge_power = self._max_charge_power(self.state[0])
        # bess_power = min(bess_power, max_charge_power)
        
        # Atualizando o SOC da bateria, garantindo que não reduza abaixo de 0 ou aumente acima de 1
        soc_change = bess_power * (self.time_step_minutes / 60) / self.BESSEmax  # Convertendo time_step de minutos para horas e normalizando
        new_soc = self.state[0] + soc_change
        soc_violation_penalty = 0
        if new_soc < self.soc_min:
            soc_violation_penalty = self.soc_violation_penalty * (self.soc_min - new_soc)
            # bess_power = -self.state[0] * self.BESSEmax / (self.time_step_minutes / 60)  # Ajuste para não permitir SoC < 0
            # new_soc = 0
        elif new_soc > self.soc_max:
            soc_violation_penalty = self.soc_violation_penalty * (new_soc - self.soc_max)
            # bess_power = (1 - self.state[0]) * self.BESSEmax / (self.time_step_minutes / 60)  # Ajuste para não permitir SoC > 1
            # new_soc = 1

        # Mantendo o SoC dentro dos limites
        new_soc = np.clip(new_soc, 0, 1)

        # Calculando a energia inicial sem cortes
        energy = bess_power - pv_power + load_power

        # Inicializando pv_shedding e load_shedding
        pv_shedding = 0
        load_shedding = 0
        grid_power = 0  # Inicializando grid_power

        # Calculando a potência da rede e ajustando para estado off-grid
        if off_grid:
            net_energy = 0
            if energy > 0:
                load_shedding = 1  # Proporção de carga cortada (todo o excesso de energia)
            else:
                pv_shedding = 1  # Corte total de PV
        else:
            if energy > 0:
                net_energy = self.eta_ac_dc * energy
            else:
                net_energy = self.eta_dc_ac * energy
            grid_power = -net_energy
            if grid_power > self.max_import_power:
                pv_shedding = 1  # Corte de PV se potência da rede for violada
        
        # Recalculando a energia após ajustes de shedding
        adjusted_energy = bess_power - pv_power * (1 - pv_shedding) + load_power * (1 - load_shedding)

        # Calculando custos e recompensas
        if off_grid:
            grid_cost = 0
            load_shedding_cost = load_shedding * load_power * self.load_shedding_penalty
        else:
            if adjusted_energy > 0:
                grid_cost = adjusted_energy * self.grid_cost_import
            else:
                grid_cost = adjusted_energy * self.grid_cost_export
            load_shedding_cost = load_shedding * load_power * self.load_shedding_penalty
        
        reward = - (grid_cost + load_shedding_cost + power_violation_penalty + soc_violation_penalty)
        
        # Atualizando o estado
        self.state = np.array([
            new_soc,                         # Posição 0: Novo SoC da bateria
            pv_power,                        # Posição 1: Potência PV (instantâneo)
            load_power,                      # Posição 2: Potência da Carga (instantâneo)
            grid_power if not off_grid else 0,  # Posição 3: Potência da Rede
            pv_power * (1 - pv_shedding),    # Posição 4: PV total após corte
            load_power,                      # Posição 5: Potência da Carga (instantâneo)
            load_power * load_shedding,      # Posição 6: Load shedding
            grid_cost + load_shedding_cost,  # Posição 7: Custo com base no time_step
            new_soc,                         # Posição 8: SoC do BESS (normalizado)
            1 if off_grid else 0,            # Posição 9: Estado da Rede (1 se off-grid, 0 se on-grid)
            bess_power                       # Posição 10: Potência da bateria
        ], dtype=np.float32)  # Garantir que todos os elementos sejam float
        
        # Verificação de término do episódio
        done = (new_soc < self.soc_min or new_soc > self.soc_max)
        done_reason = ""
        if new_soc < self.soc_min:
            done_reason = "SOC below minimum threshold"
        elif new_soc > self.soc_max:
            done_reason = "SOC above maximum threshold"
        
        # Informação adicional
        info = {
            'grid_cost': grid_cost,
            'load_shedding_cost': load_shedding_cost,
            'off_grid': off_grid,
            'import_power':adjusted_energy if adjusted_energy > 0 else 0,
            'export_power': adjusted_energy if adjusted_energy < 0 else 0,
            'done_reason': done_reason if done else ""
        }
        
        return self.state, reward, done, info

    def render(self, mode='human'):
        print(f"State: {self.state}")

# Para testar o ambiente, remova ou comente a linha abaixo quando importar no main.py
if __name__ == "__main__":
    env = NetZeroMicrogridEnv()

    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        pv_power = np.random.uniform(0, env.pv_capacity)  # Exemplo de geração PV aleatória
        load_power = np.random.uniform(0, env.demand_capacity)  # Exemplo de demanda aleatória
        action = env.action_space.sample()  # Ação aleatória
        next_state, reward, done, info = env.step(action, pv_power, load_power)
        total_reward += reward
        env.render()
    
    print(f"Total Reward: {total_reward}")
    print(f"State History: {env.state}")
    print(f"Info: {info}")
