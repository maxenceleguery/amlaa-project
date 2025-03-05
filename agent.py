import torch
import torch.nn as nn
import random

from models import DQNSolver

# DQNAgent class
class DQNAgent:
    def __init__(
        self,
        state_space,
        action_space,
        lr=1e-4,
        batch_size=64,
        gamma=0.99,
        exploration_max=1.0,
        exploration_min=0.01,
        exploration_decay=0.999,
        max_memory_size=50000,
        num_replay=1,
        target_update_freq=2000,
        model_class=DQNSolver
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_max
        self.num_replay = num_replay
        self.target_update_freq = target_update_freq

        self.memory_size = max_memory_size
        self.memory_idx = 0
        self.num_in_queue = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.local_net = model_class(state_space, action_space).to(self.device)
        self.target_net = model_class(state_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())

        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.99)
        self.loss_fn = nn.SmoothL1Loss()

        self.step = 0

        self.STATE_MEM = torch.zeros((self.memory_size, *state_space), dtype=torch.float)
        self.ACTION_MEM = torch.zeros((self.memory_size, 1), dtype=torch.long)
        self.REWARD_MEM = torch.zeros((self.memory_size, 1), dtype=torch.float)
        self.NEXT_MEM = torch.zeros((self.memory_size, *state_space), dtype=torch.float)
        self.DONE_MEM = torch.zeros((self.memory_size, 1), dtype=torch.float)

    def remember(self, state, action, reward, next_state, done):
        idx = self.memory_idx % self.memory_size
        self.STATE_MEM[idx] = state.detach().cpu()
        self.ACTION_MEM[idx] = action.detach().cpu()
        self.REWARD_MEM[idx] = torch.tensor([reward], dtype=torch.float)
        self.NEXT_MEM[idx] = next_state.detach().cpu()
        self.DONE_MEM[idx] = torch.tensor([done], dtype=torch.float)
        self.memory_idx += 1
        self.num_in_queue = min(self.num_in_queue + 1, self.memory_size)

    def recall(self):
        indices = random.sample(range(self.num_in_queue), k=self.batch_size)
        state = self.STATE_MEM[indices].to(self.device)
        action = self.ACTION_MEM[indices].to(self.device)
        reward = self.REWARD_MEM[indices].to(self.device)
        next_state = self.NEXT_MEM[indices].to(self.device)
        done = self.DONE_MEM[indices].to(self.device)
        return state, action, reward, next_state, done

    def act(self, state, evaluate=False, sample=False):
        if random.random() < self.exploration_rate and not evaluate:
            return torch.tensor([[random.randrange(self.action_space)]], device=self.device)
        with torch.no_grad():
            q_values = self.local_net(state.unsqueeze(0))
            if sample:
                probs = torch.softmax(q_values, dim=1).squeeze(0).cpu().numpy()
                action = random.choices(range(self.action_space), weights=probs)[0]
                return torch.tensor([[action]], device=self.device)
            else:
                return q_values.argmax(dim=1, keepdim=True)

    def experience_replay(self):
        if self.num_in_queue < self.batch_size:
            return
        for _ in range(self.num_replay):
            self.step += 1
            states, actions, rewards, next_states, dones = self.recall()

            # Double DQN
            with torch.no_grad():
                next_actions = self.local_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = rewards.squeeze(1) + (1 - dones.squeeze(1)) * self.gamma * next_q

            current_q = self.local_net(states).gather(1, actions.long()).squeeze(1)
            loss = self.loss_fn(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_exploration_rate()

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def update_target_network(self):
        self.target_net.load_state_dict(self.local_net.state_dict())

    def save(self, path):
        torch.save(self.local_net.state_dict(), path)

    def load(self, path):
        self.local_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.local_net.state_dict())
        self.local_net.to(self.device)
        self.target_net.to(self.device)
