import torch
import torch.nn as nn
import torch.optim as optim
import random
from models import DQNSolver, PolicyNetwork

# DQNAgent class
class DQNAgent:
    def __init__(
        self,
        state_space,
        action_space,
        max_memory_size: int = 30000,
        batch_size: int = 64,
        gamma: float = 0.9,
        lr: float = 0.00025,
        exploration_max: float = 0.90,
        exploration_min: float = 0.02,
        exploration_decay: float = 0.999,
        model=DQNSolver
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.max_memory_size = max_memory_size
        self.memory_sample_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = self.exploration_max
        self.step = 0
        self.copy = 2500

        self.STATE_MEM = torch.zeros((max_memory_size, *state_space))
        self.ACTION_MEM = torch.zeros((max_memory_size, 1))
        self.REWARD_MEM = torch.zeros((max_memory_size, 1))
        self.STATE2_MEM = torch.zeros((max_memory_size, *state_space))
        self.DONE_MEM = torch.zeros((max_memory_size, 1))
        self.ending_position = 0
        self.num_in_queue = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_net = model(state_space, action_space).to(self.device)
        self.target_net = model(state_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.99)
        self.l1 = nn.SmoothL1Loss()

    def save(self, path):
        torch.save(self.local_net.state_dict(), path)

    def load(self, path):
        self.local_net.load_state_dict(torch.load(path))
        self.local_net.to(self.device)

    def act(self, state, evaluate=False, sample=False):
        if random.random() < self.exploration_rate and not evaluate:
            return torch.tensor([[random.randrange(self.action_space)]], dtype=torch.long, device=self.device)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device).unsqueeze(0)
                out = self.local_net(state)
                if sample:
                    out = torch.softmax(out, dim=-1)
                    return torch.tensor(
                        [[random.choices(range(self.action_space), weights=out.squeeze().tolist())[0]]],
                        dtype=torch.long,
                        device=self.device
                    )
                else:
                    return out.argmax(dim=1, keepdim=True).cpu().float()

    def copy_model(self):
        self.target_net.load_state_dict(self.local_net.state_dict())

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float().clone().detach()
        self.ACTION_MEM[self.ending_position] = action.float().clone().detach()
        self.REWARD_MEM[self.ending_position] = reward.float().clone().detach()
        self.STATE2_MEM[self.ending_position] = state2.float().clone().detach()
        self.DONE_MEM[self.ending_position] = done.float().clone().detach()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def recall(self):
        idx = random.sample(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx].to(self.device)
        ACTION = self.ACTION_MEM[idx].to(self.device)
        REWARD = self.REWARD_MEM[idx].to(self.device)
        STATE2 = self.STATE2_MEM[idx].to(self.device)
        DONE = self.DONE_MEM[idx].to(self.device)
        return STATE, ACTION, REWARD, STATE2, DONE

    def experience_replay(self, num_replay: int = 1):
        for _ in range(num_replay):
            if self.step % self.copy == 0:
                self.copy_model()

            if self.memory_sample_size > self.num_in_queue:
                return

            STATE, ACTION, REWARD, STATE2, DONE = self.recall()

            self.optimizer.zero_grad()

            # Double DQN target calculation
            best_actions = self.local_net(STATE2).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(STATE2).gather(1, best_actions)
            target = REWARD + (1 - DONE) * self.gamma * next_q_values

            current = self.local_net(STATE).gather(1, ACTION.long())
            loss = self.l1(current, target.detach())
            loss.backward()
            self.optimizer.step()
            self.step += 1
            
class PolicyGradientAgent:
    def __init__(self, state_space, action_space, lr=1e-4, gamma=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PolicyNetwork(state_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.memory = []  # Stores (state, action, reward)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.to(self.device)

    def act(self, state, evaluate=False, sample=False):
        # state est déjà un np.array ou un torch.Tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        if state.ndim == 3:
            state = state.unsqueeze(0)
        
        probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)

        if evaluate:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_step(self, state, action_log_prob, reward):
        """Store state, action log probability, and reward for training."""
        self.memory.append((state, action_log_prob, reward))

    def compute_returns(self):
        """Compute discounted rewards (returns)."""
        returns = []
        G = 0
        for _, _, reward in reversed(self.memory):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def update_policy(self):
        """Perform policy gradient update using REINFORCE."""
        returns = self.compute_returns()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        loss = 0
        for (_, log_prob, _), G in zip(self.memory, returns):
            loss += -log_prob * G  # Policy Gradient Loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []  # Clear memory after update
