import os
import imageio
import numpy as np
import torch
import time

def save_video(env, agent, video_dir_path='videos', max_steps=3000):
    os.makedirs(video_dir_path, exist_ok=True)
    frames = []

    # Make sure the environment is using 'rgb_array' render_mode
    state, _ = env.reset()
    # Adjust for PolicyGradientAgent if needed
    if hasattr(agent, "device"):
        state = torch.tensor(state.__array__(), dtype=torch.float32, device=agent.device)

    for _ in range(max_steps):
        # Render frame at each step
        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame))
        
        action_out = agent.act(state, evaluate=True)
        if isinstance(action_out, tuple):
            action = action_out[0]
        else:
            action = action_out

        # Step environment
        if torch.is_tensor(action):
            action = int(action.item())
        next_state, reward, done, trunc, info = env.step(action)

        # Update state
        if hasattr(agent, "device"):
            state = torch.tensor(next_state.__array__(), dtype=torch.float32, device=agent.device)
        else:
            state = next_state

        if done or trunc:
            break

    # Save the frames to MP4 using imageio
    video_path = os.path.join(video_dir_path, f"final_run_{int(time.time())}.mp4")
    # codec='libx264' lets you control compression, etc.
    imageio.mimsave(video_path, frames, fps=10, macro_block_size=None)
    return video_path
