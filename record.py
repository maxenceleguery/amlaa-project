import os
import imageio
import numpy as np
import torch
import time 
def save_video(env, agent, video_dir_path='videos', max_steps=3000):
    os.makedirs(video_dir_path, exist_ok=True)
    frames = []
    state, _ = env.reset()
    state = torch.tensor(state.__array__(), dtype=torch.float, device=agent.device)
    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action = agent.act(state, evaluate=True)
        next_state, reward, done, trunc, info = env.step(int(action.item()))
        state = torch.tensor(next_state.__array__(), dtype=torch.float, device=agent.device)
        if done or trunc:
            break
    video_path = os.path.join(video_dir_path, f"final_run_{int(time.time())}.gif")
    imageio.mimsave(video_path, [np.array(f) for f in frames], fps=30)
    return video_path
