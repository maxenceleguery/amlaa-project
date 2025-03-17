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

class BetterRewardWrapper(gym.RewardWrapper):
    """
    Your custom reward:
      - + (x_pos - max_x) if x_pos > max_x
      - If info["flag_get"]: +500 and done
      - If life <2: -500 and done
      - If agent doesn't move horizontally for >100 steps: -500 and done
      - Then scaled by /10.
    """

    def __init__(self, env=None):
        super(BetterRewardWrapper, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return obs, info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        # Additional reward for forward movement
        if info["x_pos"] > self.max_x:
            reward += (info["x_pos"] - self.max_x)

        # If agent not moving horizontally
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0

        # If it hasn't moved for 100 steps, end the episode and penalize
        if self.current_x_count > 100:
            done = True

        # If the agent reached the flag, big bonus
        if info.get("flag_get", False):
            reward += 500
            done = True
            print("GOAL")

        # If agent lost a life, big penalty
        if info["life"] < 2:
            done = True

        # Bookkeeping
        self.current_score = info["score"]
        self.max_x = max(self.max_x, info["x_pos"])
        self.current_x = info["x_pos"]

        # Scale final reward
        return state, reward / 10.0, done, truncated, info



# ========================
# 1) Utility Functions
# ========================
def preprocess(obs: np.ndarray, color: bool = True) -> np.ndarray:
    import cv2
    # Redimensionnement à 84x84
    obs = cv2.resize(obs, (84, 84))
    if color:
        # Conversion en float et normalisation
        obs = obs.astype(np.float32) / 255.0
        # Passage de (H,W,C) à (C,H,W)
        obs = np.transpose(obs, (2, 0, 1))
    else:
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = obs.astype(np.float32) / 255.0
        obs = np.expand_dims(obs, axis=0)
    return obs

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

        action = agent.act(state_t, evaluate=False)
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
# Autoencodeur convolutionnel pour images couleur
class ConvAutoEncoder(nn.Module):
    def __init__(self, latent_dim=256, in_channels=3, dropout_prob=0.5):
        super(ConvAutoEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # (32,42,42)
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # (64,21,21)
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # (128,10,10)
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),         # (256,5,5)
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(),
        )
        # 256*5*5 = 6400
        self.enc_fc = nn.Linear(256 * 5 * 5, latent_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256 * 5 * 5)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128,10,10)
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (64,20,20)
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (32,40,40)
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # (3,80,80)
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(x.size(0), -1)  # (B, 6400)
        z = self.enc_fc(h)
        z = self.dropout(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z)
        h = h.view(z.size(0), 256, 5, 5)
        x_recon = self.decoder(h)            # (B, 3, 80, 80)
        # Interpolation bilinéaire pour obtenir (3,84,84)
        x_recon = F.interpolate(x_recon, size=(84, 84), mode='bilinear')
        return x_recon

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        out = self.decode(z)
        return out, z
    
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
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

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
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        # On stocke un compteur pour chaque (level, x_pos)
        # plutôt qu'un booléen
        self.seen_positions = {}  # (level_name, x_pos)-> count

    def add(self, img: np.ndarray, level: str, x_pos: int):
        key = (level, x_pos)
        count = self.seen_positions.get(key, 0)
        # # On autorise jusqu'à 2 enregistrements max
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
                                   replay_buffer,  # supposez que c'est votre dataset sous forme de replay buffer
                                   device="cpu", batch_size=64,
                                   lr=1e-3, epochs=5, lambda_l2=1e-2):
    from torch.utils.data import DataLoader

    ds = MarioImageDataset(replay_buffer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
    mse_loss = nn.MSELoss()

    final_loss = 0.0
    for ep in range(epochs):
        total_loss = 0.0
        count = 0
        for imgs_np in loader:
            imgs_t = imgs_np.float().to(device)  # (B, C, 84,84)
            recon, z = autoencoder(imgs_t)
            loss_recon = mse_loss(recon, imgs_t)
            # Pénalisation L2 sur le latent pour favoriser la parcimonie
            loss_l2 = lambda_l2 * torch.mean(z**2)
            loss = loss_recon + loss_l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
        mean_loss = total_loss / count
        print(f"AE epoch {ep+1}/{epochs}, mean loss={mean_loss:.4f}")
        final_loss = mean_loss

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
        # Assurez-vous d'appeler la méthode forward qui retourne (reconstruction, latent)
        recons_t, _ = autoencoder(imgs_t)

    for i in range(n):
        orig_img = imgs_t[i].cpu().numpy()  # forme attendue : (C,84,84)
        recon_img = recons_t[i].cpu().numpy()  # idem

        # Si l'image est en niveaux de gris (1 canal), on supprime l'axe du canal.
        if orig_img.shape[0] == 1:
            orig_img = np.squeeze(orig_img, axis=0)
        # Pour une image couleur (3 canaux), on transpose pour obtenir HWC
        elif orig_img.shape[0] == 3:
            orig_img = np.transpose(orig_img, (1, 2, 0))
            
        if recon_img.shape[0] == 1:
            recon_img = np.squeeze(recon_img, axis=0)
        elif recon_img.shape[0] == 3:
            recon_img = np.transpose(recon_img, (1, 2, 0))

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
        replay_buffer = ImageReplayBuffer(capacity=10000)

    # Rolling average init
    rolling_env_avg = {lvl: 1.0 for lvl in levels}

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
        best_ep_return = -float("inf")
        best_frames_info_list = None
        
        # Evaluate each solution in the population.
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
            
            # Update rolling average reward for the level.
            old_avg = rolling_env_avg[lvl]
            new_avg = 0.9 * old_avg + 0.1 * ep_return
            rolling_env_avg[lvl] = new_avg

            gen_rewards.append(ep_return)
            fitnesses.append(-ep_return)
            
            # Keep only the best individual's rollout frames.
            if ep_return >= best_ep_return*0.9:
                best_ep_return = ep_return
                best_frames_info_list = frames_info_list

        # Add only the best individual's rollout frames to the replay buffer.
        if best_frames_info_list is not None:
            replay_buffer.add_batch(best_frames_info_list)

        mean_r = np.mean(gen_rewards)
        best_f = es.result.fbest

        # Autoencoder learning rate decay.
        lr_current = lr_init * (lr_decay ** gen)
        ae_loss = train_autoencoder_full_dataset(
            autoencoder, replay_buffer,
            device=device, batch_size=64, lr=lr_current, epochs=ae_epochs
        )
        if ae_loss is None:
            ae_loss = 0.0

        es.tell(solutions, fitnesses)
        es.logger.add()

        # Log AE reconstructions.
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

        # Video logging after each generation.
        best_sol = es.result.xbest
        best_f_sol = es.result.fbest
        # Create a best policy.
        best_policy = LatentPolicy(num_actions=7, latent_dim=policy.latent_dim, hidden_dim=policy.hidden_dim).to(device)
        best_policy.set_params_vector(best_sol)
        best_agent = MarioCmaAgent(best_policy, autoencoder, device=device)

        # Pick a random level for video logging.
        lvl_vid = random.choice(levels)
        env_vid = make_mario_env(lvl_vid, max_steps=5000, render=True)
        
        # Only record video if current best fitness is better than the previous best.
        if best_f < es.result.fbest:
            video_path = save_video(env_vid, best_agent, video_dir_path='videos', max_steps=5000)
            wandb.log({f"video_gen_{gen}": wandb.Video(video_path)})
            
        env_vid.close()

    final_sol = es.result.xbest
    final_f = es.result.fbest
    policy.set_params_vector(final_sol)
    return policy, final_f


def main():
    wandb.init(project="MarioCMAES-ImprovedReward", name="BetterReward_IncrementalEnv", config={
        "latent_dim": 128,
        "popsize": 30,
        "nb_generations": 500,
        "ae_epochs": 10,
        "hidden_dim": 32,
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
    policy = LatentPolicy(latent_dim=config.latent_dim, num_actions=nA, hidden_dim=config.hidden_dim).to(device)
    agent = MarioCmaAgent(policy, autoencoder, device=device)

    # 4) replay buffer
    replay_buffer = ImageReplayBuffer(capacity=10000)

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
    replay_buffer = ImageReplayBuffer(capacity=10000)
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
    replay_buffer = ImageReplayBuffer(capacity=10000)
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
