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
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.99)
        self.l1 = nn.SmoothL1Loss()

    def save(self, path):
        torch.save(self.local_net.state_dict(), path)

    def load(self, path):
        self.local_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.local_net.state_dict())

    def update_exploration_rate(self):
        """Decay exploration rate"""
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)


    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float().clone().detach()
        self.ACTION_MEM[self.ending_position] = action.float().clone().detach()
        self.REWARD_MEM[self.ending_position] = reward.float().clone().detach()
        self.STATE2_MEM[self.ending_position] = state2.float().clone().detach()
        self.DONE_MEM[self.ending_position] = done.float().clone().detach()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO buffer
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

        
    def recall(self):
        # Randomly sample 'batch size' experiences
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

            # Wait until enough samples are in queue
            if self.memory_sample_size > self.num_in_queue:
                return

            STATE, ACTION, REWARD, STATE2, DONE = self.recall()
            
            self.optimizer.zero_grad()
            # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
            target = REWARD + torch.mul((self.gamma * self.target_net(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)

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

    def act(self, state, evaluate=False, sample=False):
        """Sample an action from the policy."""
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

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
