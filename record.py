import os
import time
import pygame
import numpy as np
import cv2
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def save_video(env, agent=None, video_dir_path='mario_videos_test', max_steps=2000):
    # Create video directory if it doesn't exist
    os.makedirs(video_dir_path, exist_ok=True)

    # Initialize pygame for display
    pygame.init()
    clock = pygame.time.Clock()

    # Create the environment
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Reset the environment
    try:
        state, info = env.reset()  # Try newer API first
    except ValueError:
        state = env.reset()
        info = {}

    # Get initial frame to set up display
    frame = env.render()
    if frame is None:
        raise ValueError("Could not get initial frame")

    # Set up pygame display
    screen_width, screen_height = frame.shape[1], frame.shape[0]
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Super Mario Bros')

    # Set up video writer
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if agent is not None:
        video_path = os.path.join(video_dir_path, f'{agent.name}-{timestamp}.mp4')
    else:
        video_path = os.path.join(video_dir_path, f'random-{timestamp}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30.0, (screen_width, screen_height))

    # Game loop variables
    steps = 0
    done = False
    total_reward = 0

    if agent is not None:
        print(f"Agent: {agent}")
    else:
        print("Agent: Random")
    try:
        while steps < max_steps and not done:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            # Take a random action
            if agent is not None:
                action = agent.act(state)
            else:
                action = env.action_space.sample()
            
            # Step the environment
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError:
                obs, reward, done, info = env.step(action)
            
            # Get frame from environment
            frame = env.render()
            
            if frame is not None:
                # Display frame using pygame
                pygame_surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
                screen.blit(pygame_surface, (0, 0))
                pygame.display.flip()
                
                # Save frame to video (convert from RGB to BGR for OpenCV)
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            
            # Update metrics
            total_reward += reward
            steps += 1
            
            # Print progress
            if steps % 100 == 0:
                print(f"Step: {steps}, Position: {info.get('x_pos', 'N/A')}, Time: {info.get('time', 'N/A')}")
            
            # End conditions
            if done or info.get('time', 400) <= 100:
                break
                
            # Control frame rate
            clock.tick(30)
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Release video writer
        video.release()
        
        # Close pygame
        pygame.quit()
        
        # Close environment
        try:
            env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")

    print(f"Episode finished after {steps} steps with total reward {total_reward}")
    print(f"Video saved to {video_path}")