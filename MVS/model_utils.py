import torch
import random
from collections import deque, namedtuple

class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_dim)

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

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()
