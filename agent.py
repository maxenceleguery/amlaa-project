import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from models import DQNSolver, PolicyNetwork, RainbowDQN

class ReplayBuffer:
    def __init__(self, state_dim, max_size):
        self.max_size = max_size
        self.position = 0
        self.count = 0
        self.states = torch.zeros((max_size, *state_dim))
        self.actions = torch.zeros((max_size, 1), dtype=torch.long)
        self.rewards = torch.zeros((max_size, 1))
        self.next_states = torch.zeros((max_size, *state_dim))
        self.dones = torch.zeros((max_size, 1))

    def store(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)

    def sample(self, batch_size, device):
        indices = random.sample(range(self.count), batch_size)
        return (
            self.states[indices].to(device),
            self.actions[indices].to(device),
            self.rewards[indices].to(device),
            self.next_states[indices].to(device),
            self.dones[indices].to(device),
        )

class SumTree:
    """Binary tree-based SumTree for fast prioritized sampling"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # SumTree nodes
        self.data = np.zeros(capacity, dtype=object)  # Experience buffer
        self.position = 0  # Current insert position

    def add(self, priority, experience):
        """Add experience with priority"""
        index = self.position + self.capacity - 1
        self.data[self.position] = experience  # Store experience
        self.update(index, priority)  # Update tree
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def update(self, index, priority):
        """Update priority and propagate change"""
        change = priority - self.tree[index]
        self.tree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += change  # Update parents

    def get_leaf(self, value):
        """Get the leaf node (sample an experience)"""
        parent_idx = 0
        while parent_idx < self.capacity - 1:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1
            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child
        data_idx = parent_idx - (self.capacity - 1)
        return parent_idx, self.tree[parent_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]  # Root node contains total sum

class PrioritizedReplayBuffer:
    def __init__(self, state_dim, max_size, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(max_size)
        self.alpha = alpha  # Prioritization factor
        self.beta = beta  # Importance sampling correction factor
        self.beta_increment = beta_increment
        self.max_size = max_size
        self.count = 0
        self.epsilon = 1e-5  # Small constant for stability

    def store(self, state, action, reward, next_state, done):
        """Store experience with max priority initially"""
        max_priority = max(self.tree.tree[-self.tree.capacity:])  # Max priority
        if max_priority == 0:
            max_priority = 1.0  # Default initial priority
        experience = (state, action, reward, next_state, done)
        self.tree.add(max_priority, experience)  # Store with max priority
        self.count = min(self.count + 1, self.max_size)

    def sample(self, batch_size, device):
        """Sample batch with probability proportional to priority"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size  # Split tree into segments

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            index, priority, experience = self.tree.get_leaf(value)
            batch.append(experience)
            indices.append(index)
            priorities.append(priority)

        self.beta = min(1.0, self.beta + self.beta_increment)  # Increase β
        sampling_probs = np.array(priorities) / self.tree.total_priority()
        is_weights = (self.count * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()  # Normalize

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.stack(actions).to(device),
            torch.stack(rewards).to(device),
            torch.stack(next_states).to(device),
            torch.stack(dones).to(device).to(dtype=torch.long),
            torch.tensor(is_weights, device=device, dtype=torch.float32),
            indices
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD error"""
        priorities = (td_errors.abs().cpu().numpy() + self.epsilon) ** self.alpha
        for i, index in enumerate(indices):
            self.tree.update(index, priorities[i])


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
        self.replay_buffer_sample_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = self.exploration_max
        self.step = 1
        self.copy = 2500

        self.replay_buffer = PrioritizedReplayBuffer(state_space, max_memory_size)

        """
        self.STATE_MEM = torch.zeros((max_memory_size, *state_space))
        self.ACTION_MEM = torch.zeros((max_memory_size, 1))
        self.REWARD_MEM = torch.zeros((max_memory_size, 1))
        self.STATE2_MEM = torch.zeros((max_memory_size, *state_space))
        self.DONE_MEM = torch.zeros((max_memory_size, 1))
        self.ending_position = 0
        self.num_in_queue = 0
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_net = model(state_space, action_space).to(self.device)
        self.target_net = model(state_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=lr)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.995)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.999, patience=5, verbose=True)
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

    def copy_model(self, tau=0.1):  # Soft update
        #self.target_net.load_state_dict(self.local_net.state_dict())
        for target_param, local_param in zip(self.target_net.parameters(), self.local_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        print("Model updated !")

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def remember(self, state, action, reward, state2, done):
        self.replay_buffer.store(state, action, reward, state2, done)

    def recall(self):
        return self.replay_buffer.sample(self.replay_buffer_sample_size, self.device)

    def experience_replay(self, num_replay: int = 1):
        for _ in range(num_replay):
            #if self.step % self.copy == 0:
            #    self.copy_model()

            if self.replay_buffer_sample_size > self.replay_buffer.count:
                return

            STATE, ACTION, REWARD, STATE2, DONE, is_weights, indices = self.recall()

            self.optimizer.zero_grad()

            # Double DQN target calculation
            with torch.no_grad():
                best_actions = self.local_net(STATE2).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(STATE2).gather(1, best_actions)
                target = REWARD + (1 - DONE) * self.gamma * next_q_values

            out = self.local_net(STATE)
            if ACTION.dim() > 2:
                ACTION = ACTION.squeeze(-1)
            current_q = out.gather(1, ACTION.long())
            td_errors = target.detach() - current_q
            loss = (is_weights * td_errors ** 2).mean()

            #loss = self.l1(current_q, target.detach())
            loss.backward()
            self.optimizer.step()
            self.step += 1

            #elf.replay_buffer.update_priorities(indices, td_errors)
            
class PolicyGradientAgent:
    def __init__(self, state_space, action_space, lr=1e-4, gamma=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PolicyNetwork(state_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.replay_buffer = []  # Stores (state, action, reward)

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
        self.replay_buffer.append((state, action_log_prob, reward))

    def compute_returns(self):
        """Compute discounted rewards (returns)."""
        returns = []
        G = 0
        for _, _, reward in reversed(self.replay_buffer):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def update_policy(self):
        """Perform policy gradient update using REINFORCE."""
        returns = self.compute_returns()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        loss = 0
        for (_, log_prob, _), G in zip(self.replay_buffer, returns):
            loss += -log_prob * G  # Policy Gradient Loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.replay_buffer = []  # Clear memory after update


class MultiStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) == self.n_step:
            return self._get_n_step_transition()
        return None

    def _get_n_step_transition(self):
        """Compute n-step reward & return transition"""
        state, action = self.buffer[0][:2]
        next_state, done = self.buffer[-1][-2:]
        
        R = sum(self.gamma**i * self.buffer[i][2] for i in range(self.n_step))
        self.buffer.pop(0)
        return state, action, R, next_state, done

    def reset(self):
        self.buffer.clear()

class RainbowAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, atom_size=51, v_min=-10, v_max=10, lr=0.00025):
        self.step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.n_step = 3
        self.replay_buffer = PrioritizedReplayBuffer(state_dim, 30000)  # ✅ Use PER
        self.multi_step_buffer = MultiStepBuffer(n_step=self.n_step, gamma=self.gamma)

        self.local_net = RainbowDQN(state_dim, action_dim, atom_size, v_min, v_max).to(self.device)
        self.target_net = RainbowDQN(state_dim, action_dim, atom_size, v_min, v_max).to(self.device)
        self.target_net.load_state_dict(self.local_net.state_dict())
        self.optimizer = optim.Adam(self.local_net.parameters(), lr=lr)

    def experience_replay(self, batch_size=64):
        states, actions, rewards, next_states, dones, is_weights, indices = self.replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_q_dist = self.target_net(next_states)
            next_q = (next_q_dist * self.local_net.support).sum(dim=-1)
            next_actions = next_q.argmax(dim=1, keepdim=True)
            next_q_values = next_q_dist.gather(1, next_actions.unsqueeze(-1)).squeeze()

        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_dist = self.local_net(states)
        current_q = current_q_dist.gather(1, actions.unsqueeze(-1)).squeeze()

        loss = (is_weights * (target_q - current_q) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1

        self.replay_buffer.update_priorities(indices, (target_q - current_q).abs())

    def reset_noise(self):
        """Resample noise in NoisyNet"""
        self.local_net.reset_noise()
