import os
import sys
import time
import math
import random
import logging
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import TimeLimit

import cma
import wandb
import imageio

# ========================
# Configuration du Logging
# ========================
logger = logging.getLogger("Train_CMAES_Autoencoder")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# ========================
# 0) Custom Reward Wrapper
# ========================
class BetterRewardWrapper(gym.RewardWrapper):
    """
    A simple reward wrapper that provides reward only at the final step:
      final_reward = (info["x_pos"] - initial_x_pos_of_episode)
    Intermediate steps return 0 reward.
    """
    def __init__(self, env=None):
        super(BetterRewardWrapper, self).__init__(env)
        self.init_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Record the initial x-position at the start of an episode
        self.init_x = info.get("x_pos", 0)
        return obs, info

    def step(self, action):
        state, _, done, truncated, info = self.env.step(action)

        if done or truncated:
            # Final reward is last_x_pos - init_x
            reward = info.get("x_pos", 0) - self.init_x
        else:
            # No intermediate reward
            reward = 0.0

        return state, reward, done, truncated, info



# ========================
# 1) Utility Functions
# ========================
def preprocess(obs: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale, resize to (84,84), scale [0,1].
    Output shape: (1,84,84).
    """
    import cv2
    gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (84,84))
    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=0)  # (1,84,84)
    return gray

def make_mario_env(level='1-1', max_steps=3000, render=False):
    """
    Creates a SuperMarioBros environment with your BetterRewardWrapper for a custom reward.
    """
    render_mode = 'rgb_array' if render else 'none'
    env_id = f"SuperMarioBros-{level}-v0"
    base_env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, render_mode=render_mode)
    env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    env = TimeLimit(env, max_episode_steps=max_steps)
    # Wrap with the custom reward:
    env = BetterRewardWrapper(env)
    return env

def save_video(env, agent, video_dir_path='videos', max_steps=3000):
    """
    Record a video of the agent in the environment.
    """
    os.makedirs(video_dir_path, exist_ok=True)
    frames = []

    state, info = env.reset()
    state_t = torch.tensor(preprocess(state).copy(), dtype=torch.float32, device=agent.device)

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame))

        action = agent.act(state_t, evaluate=True)
        next_state, reward, done, trunc, info = env.step(action)
        next_state_t = torch.tensor(preprocess(next_state).copy(), dtype=torch.float32, device=agent.device)
        state_t = next_state_t

        if done or trunc:
            break

    video_path = os.path.join(video_dir_path, f"final_run_{int(time.time())}.mp4")
    imageio.mimsave(video_path, frames, fps=25, macro_block_size=None)
    return video_path


# ========================
# 2) AE with BatchNorm
# ========================
class ConvAutoEncoder(nn.Module):
    """
    AE for 1×84×84 images, latent=128, uses BN. 
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8,momentum=0.1),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16,momentum=0.1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32,momentum=0.1),
            nn.ReLU(),
        )
        # After 3 convs => (32,10,10), flatten=3200
        self.enc_fc = nn.Linear(32*10*10, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 32*10*10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16,momentum=0.1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8,momentum=0.1),
            nn.ReLU(),

            # final: kernel=8 => (40->84)
            nn.ConvTranspose2d(8, 1, kernel_size=8, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # (B,3200)
        z = self.enc_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z)
        h = h.view(h.size(0), 32, 10, 10)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


# ========================
# 3) Policy
# ========================
class LatentPolicy(nn.Module):
    """
    Softmax policy with a layernorm of the latent input.
    """
    def __init__(self, latent_dim=128, num_actions=7, hidden_dim=64):
        super().__init__()
        self.ln = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, z: torch.Tensor, evaluate=False) -> int:
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        #z = self.ln(z)
        h = F.relu(self.fc1(z))
        logits = self.fc2(h)
        probs = F.softmax(logits, dim=1)
        if evaluate:
            return probs.argmax(dim=1).item()
        else:
            dist = torch.distributions.Categorical(probs=probs)
            return dist.sample().item()

    def get_params_vector(self) -> np.ndarray:
        with torch.no_grad():
            return nn.utils.parameters_to_vector(self.parameters()).cpu().numpy()

    def set_params_vector(self, np_params: np.ndarray):
        t = torch.tensor(np_params, dtype=torch.float32, device=next(self.parameters()).device)
        nn.utils.vector_to_parameters(t, self.parameters())


# ========================
# CMA Agent
# ========================
class MarioCmaAgent:
    def __init__(self, policy: LatentPolicy, autoencoder: ConvAutoEncoder, device="cpu"):
        self.policy = policy
        self.ae = autoencoder
        self.device = device

    def act(self, state: torch.Tensor, evaluate=True) -> int:
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        with torch.no_grad():
            z = self.ae.encode(state)
        return self.policy.forward(z[0], evaluate=evaluate)

# ========================
# 4) Replay Buffer
# ========================
class ImageReplayBuffer:
    """
    On stocke jusqu'à 2 images pour chaque (level_name, x_pos)
    pour plus de diversité.
    """
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        # On stocke un compteur pour chaque (level, x_pos)
        # plutôt qu'un booléen
        self.seen_positions = {}  # (level_name, x_pos)-> count

    def add(self, img: np.ndarray, level: str, x_pos: int):
        key = (level, x_pos)
        count = self.seen_positions.get(key, 0)
        # # On autorise jusqu'à 2 enregistrements max

        self.seen_positions[key] = count + 1
        self.buffer.append(img)

    def add_batch(self, frames_info_list):
        for (img, lvl, xpos) in frames_info_list:
            self.add(img, lvl, xpos)

    def sample(self, batch_size: int) -> np.ndarray:
        if len(self.buffer) < batch_size:
            return None
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        return np.array([self.buffer[i] for i in idxs])

    def __len__(self):
        return len(self.buffer)


class MarioImageDataset(Dataset):
    def __init__(self, buffer: ImageReplayBuffer):
        self.buffer = buffer.buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]  # shape (1,84,84)


def train_autoencoder_full_dataset(autoencoder: ConvAutoEncoder,
                                   replay_buffer: ImageReplayBuffer,
                                   device="cpu", batch_size=64,
                                   lr=1e-3, epochs=5):
    """
    Train the AE on the entire replay buffer, for 'epochs' passes.
    """
    if len(replay_buffer) < batch_size:
        return None

    ds = MarioImageDataset(replay_buffer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    autoencoder.train()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)

    loss_fn = nn.MSELoss()

    final_loss = 0.0
    for ep in range(epochs):
        total_loss = 0.0
        count = 0
        for imgs_np in loader:
            imgs_t = imgs_np.float().to(device)  # (B,1,84,84)
            recon_t = autoencoder(imgs_t)
            loss = loss_fn(recon_t, imgs_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
        mean_loss = total_loss / count
        final_loss = mean_loss
        logger.info(f"AE epoch {ep+1}/{epochs}, mean loss={mean_loss:.4f}")

    autoencoder.eval()
    return final_loss


def log_ae_reconstructions(autoencoder: ConvAutoEncoder,
                           replay_buffer: ImageReplayBuffer,
                           device="cpu", n=4):
    import wandb
    if len(replay_buffer) < n:
        return

    idxs = np.random.choice(len(replay_buffer.buffer), n, replace=False)
    data_list = [replay_buffer.buffer[i] for i in idxs]
    imgs_t = torch.tensor(np.array(data_list), dtype=torch.float32, device=device)
    with torch.no_grad():
        recons_t = autoencoder(imgs_t)

    for i in range(n):
        orig_img = imgs_t[i].cpu().numpy()  # (1,84,84)
        recon_img = recons_t[i].cpu().numpy()
        orig_img = np.squeeze(orig_img, axis=0)
        recon_img = np.squeeze(recon_img, axis=0)

        wandb.log({
            f"reconstruction_sample_{i}": [
                wandb.Image(orig_img, caption="Original"),
                wandb.Image(recon_img, caption="Reconstruction")
            ]
        })


def rollout_env_collect(env, policy, autoencoder, device="cpu", max_steps=5000,
                        no_progress_skip=True, frame_skip=4, level_name="1-1"):
    """
    Plays 1 episode => total_reward & frames_info_list (img, level_name, x_pos).
    """
    obs, info = env.reset()
    total_reward = 0.0
    frames_info_list = []

    for t in range(max_steps):
        img = preprocess(obs)  # (1,84,84)
        with torch.no_grad():
            img_t = torch.tensor(img[None,:], device=device)
            z = autoencoder.encode(img_t)
            action_idx = policy.forward(z[0], evaluate=False)

        obs, reward, done, trunc, info = env.step(action_idx)
        total_reward += reward
        new_xpos = info.get('x_pos', 0)

        if t % frame_skip == 0:
            # If you want to skip frames only if progress is made, you can check new_xpos>old_xpos
            # but for now it's just a uniform sampling
            frames_info_list.append((preprocess(obs), level_name, int(new_xpos)))

        if done or trunc:
            break

    return total_reward, frames_info_list, new_xpos


def pick_env_by_low_reward(levels, rolling_env_avg):
    """
    Weighted pick: probability ~ 1/(avg+eps).
    """
    eps = 1e-3
    weights = []
    for lvl in levels:
        w = 1.0 / (rolling_env_avg[lvl] + eps)
        weights.append(w)
    sum_w = sum(weights)
    r = random.random() * sum_w
    cum = 0.0
    for lvl, w in zip(levels, weights):
        cum += w
        if r <= cum:
            return lvl
    return levels[-1]


def cma_es_loop(policy: LatentPolicy,
                autoencoder: ConvAutoEncoder,
                device="cpu",
                levels=['1-1'],
                popsize=25,             # a bit more population
                nb_generations=100,     # 100 generations
                replay_buffer=None,
                ae_epochs=10,           # 5 epochs for AE
                no_progress_skip=True,
                frame_skip=4,
                lr_init=5e-3,
                lr_decay=0.99):
    """
    CMA-ES loop with environment importance sampling (lowest rolling average reward).
    We'll do a video logging after each gen.
    """
    import wandb
    if replay_buffer is None:
        replay_buffer = ImageReplayBuffer(capacity=20000)

    # Rolling average init
    rolling_env_avg = {lvl: 2.0 for lvl in levels}

    init_params = policy.get_params_vector()
    es = cma.CMAEvolutionStrategy(init_params, 3.0, {
        'popsize': popsize,
        'seed': 1234,
        'verbose': 0
    })

    for gen in range(nb_generations):
        solutions = es.ask()
        fitnesses = []
        gen_rewards = []
        max_epochs = []
        for sol in solutions:
            policy.set_params_vector(sol)
            lvl = pick_env_by_low_reward(levels, rolling_env_avg)

            env = make_mario_env(lvl, max_steps=5000, render=False)
            ep_return, frames_info_list, max_epoch = rollout_env_collect(
                env, policy, autoencoder, device=device, 
                max_steps=5000, no_progress_skip=no_progress_skip, 
                frame_skip=frame_skip, level_name=lvl
            )
            env.close()
            max_epochs.append(max_epoch)
            # update rolling env
            old_avg = rolling_env_avg[lvl]
            new_avg = 0.9*old_avg + 0.1*ep_return
            rolling_env_avg[lvl] = new_avg

            gen_rewards.append(ep_return)
            fitnesses.append(-ep_return)
            replay_buffer.add_batch(frames_info_list)

        mean_r = np.mean(gen_rewards)
        best_f = es.result.fbest

        # autoencoder LR decays slightly
        lr_current = lr_init * (lr_decay ** gen)
        ae_loss = train_autoencoder_full_dataset(
            autoencoder, replay_buffer,
            device=device, batch_size=64, lr=lr_current, epochs=ae_epochs
        )
        if ae_loss is None:
            ae_loss = 0.0

        es.tell(solutions, fitnesses)
        es.logger.add()

        # log recon
        log_ae_reconstructions(autoencoder, replay_buffer, device=device, n=4)

        logger.info(f"[Gen {gen}] MeanReward={mean_r:.2f}, bestF={best_f:.2f}, AE_loss={ae_loss:.4f}, lrAE={lr_current:.5f}")
        wandb.log({
            "gen": gen,
            "mean_reward_gen": mean_r,
            "cma_best_fitness": best_f,
            "ae_loss": ae_loss,
            "lr_ae": lr_current,
            "max_last__epoch": max(max_epochs),
            "mean_last_epoch": np.mean(max_epochs)
        })

        # Video logging after each gen
        best_sol = es.result.xbest
        best_f_sol = es.result.fbest
        if gen % 5 == 0:

            # create a best policy
            best_policy = LatentPolicy(latent_dim=policy.ln.normalized_shape[0], num_actions=7, hidden_dim=64).to(device)
            best_policy.set_params_vector(best_sol)
            best_agent = MarioCmaAgent(best_policy, autoencoder, device=device)

            # pick random env for the video
            lvl_vid = random.choice(levels)
            env_vid = make_mario_env(lvl_vid, max_steps=5000, render=True)
            video_path = save_video(env_vid, best_agent, video_dir_path='videos', max_steps=5000)
            env_vid.close()

    final_sol = es.result.xbest
    final_f = es.result.fbest
    policy.set_params_vector(final_sol)
    return policy, final_f


def main():
    wandb.init(project="MarioCMAES-ImprovedReward", name="BetterReward_IncrementalEnv", config={
        "latent_dim": 128,
        "popsize": 20,
        "nb_generations": 500,
        "ae_epochs": 10,
        "frame_skip": 4,
        "no_progress_skip": True,
        "lr_init": 1e-3,
        "lr_decay": 0.98
    })
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 1) Create AE
    autoencoder = ConvAutoEncoder(latent_dim=config.latent_dim).to(device)

    # 2) get action space
    env_test = make_mario_env('1-1', max_steps=5000, render=False)
    nA = env_test.action_space.n
    env_test.close()

    # 3) policy
    policy = LatentPolicy(latent_dim=config.latent_dim, num_actions=nA, hidden_dim=64).to(device)
    agent = MarioCmaAgent(policy, autoencoder, device=device)

    # 4) replay buffer
    replay_buffer = ImageReplayBuffer(capacity=20000)

    # We'll do an incremental approach: first "1-1", then "1-2", then "2-1", etc.
    all_levels = []

    # Stage 1:
    all_levels = ["1-1"]
    policy, best_f = cma_es_loop(
        policy=policy, autoencoder=autoencoder, device=device,
        levels=all_levels, popsize=config.popsize, nb_generations=config.nb_generations,
        replay_buffer=replay_buffer,
        ae_epochs=config.ae_epochs,
        no_progress_skip=config.no_progress_skip,
        frame_skip=config.frame_skip,
        lr_init=config.lr_init,
        lr_decay=config.lr_decay
    )
    wandb.log({"best_f_stage1": best_f})

    # Stage 2: add "1-2"
    all_levels = ["1-1", "1-2"]
    # optionally reset replay buffer or not
    replay_buffer = ImageReplayBuffer(capacity=20000)
    policy, best_f = cma_es_loop(
        policy=policy, autoencoder=autoencoder, device=device,
        levels=all_levels, popsize=config.popsize, nb_generations=config.nb_generations,
        replay_buffer=replay_buffer,
        ae_epochs=config.ae_epochs,
        no_progress_skip=config.no_progress_skip,
        frame_skip=config.frame_skip,
        lr_init=config.lr_init,
        lr_decay=config.lr_decay
    )
    wandb.log({"best_f_stage2": best_f})

    # Stage 3: add "2-1"
    all_levels = ["1-1", "1-2", "2-1"]
    replay_buffer = ImageReplayBuffer(capacity=20000)
    policy, best_f = cma_es_loop(
        policy=policy, autoencoder=autoencoder, device=device,
        levels=all_levels, popsize=config.popsize, nb_generations=config.nb_generations,
        replay_buffer=replay_buffer,
        ae_epochs=config.ae_epochs,
        no_progress_skip=config.no_progress_skip,
        frame_skip=config.frame_skip,
        lr_init=config.lr_init,
        lr_decay=config.lr_decay
    )
    wandb.log({"best_f_stage3": best_f})

    logger.info("All stages done. Testing final policy on 2-1 or 2-2...")

    # final test
    env_final = make_mario_env('2-1', max_steps=5000, render=True)
    total_r, frames_info,max_epoch = rollout_env_collect(
        env_final, policy, autoencoder, 
        device=device, max_steps=5000, 
        no_progress_skip=False, level_name='2-1'
    )
    env_final.close()
    logger.info(f"Final test return: {total_r}")
    wandb.log({"final_return_2-1": total_r})

    # final video
    video_path = os.path.join("videos", f"final_run_{int(time.time())}.mp4")
    os.makedirs("videos", exist_ok=True)
    env_vid = make_mario_env('2-1', render=True)
    video_path = save_video(env_vid, agent, video_dir_path='videos', max_steps=1000)
    env_vid.close()

    wandb.log({"final_video_stage3": wandb.Video(video_path)})
    wandb.finish()
    logger.info("Done!")

if __name__ == "__main__":
    main()
