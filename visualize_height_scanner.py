#!/usr/bin/env python3
"""Visualization script for Go1 with height scanner."""

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import cv2
from mujoco_playground._src import registry

def main():
    # Set up visualization options
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True   # Show visual geoms
    scene_option.geomgroup[3] = False  # Hide collision geoms  
    scene_option.geomgroup[5] = True   # Show sites (including height scanner visualization)
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # Show contact points
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
    print("Creating Go1 Height Scanner Visualization...")
    
    # Load environment
    env_name = 'Go1JoystickRoughTerrain'
    print(f"Loading environment: {env_name}")
    
    # Get default config and modify as needed
    env_cfg = registry.get_default_config(env_name)
    env_cfg.episode_length = 10  # 500 steps for demo
    
    env = registry.load(env_name, config=env_cfg)
    print("✓ Environment loaded successfully!")
    
    # JIT compile the functions for speed
    print("JIT compiling functions...")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    print("✓ JIT compilation complete!")
    
    # Initialize
    key = jax.random.PRNGKey(42)
    
    print("Running simulation...")
    
    rollout = []
    
    # Reset environment
    state = jit_reset(key)
    rollout.append(state)
    
    # Run simulation with static/falling robot
    for i in range(env_cfg.episode_length):
        print(f"Step {i+1}/{env_cfg.episode_length}")
        # Zero action (let robot fall/be static)
        action = jp.zeros(env.action_size)
        
        # Step simulation
        state = jit_step(state, action)
        # print(state.obs['privileged_state'])
        rollout.append(state)
        
        # Print height scanner data every 50 steps
        if i % 1 == 0:
            try:
                height_map = env.get_height_map_rangefinder(state.data)
                height_grid = height_map.reshape(5, 5)
                print(height_grid)
                # print(f"Step {i:3d}: Height range [{np.min(height_map):.3f}, {np.max(height_map):.3f}]m")
            except Exception as e:
                print(f"Step {i:3d}: Could not get height map: {e}")
    
    print(f"✓ Simulation completed with {len(rollout)} frames")
    
    # Render video
    print("Rendering video...")
    
    render_every = 2  # Render every 2nd frame
    fps = 1.0 / env.dt / render_every
    
    traj = rollout[::render_every]
    

    
    print(f"Rendering {len(traj)} frames at {fps:.1f} fps...")
    
    frames = env.render(
        traj,
        camera="track",  # Use tracking camera
        scene_option=scene_option,
        width=640,
        height=480,
    )
    
    # Save video
    video_filename = 'go1_height_scanner_demo.mp4'
    print(f"Saving video to {video_filename}...")
    
    # Create cv2 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (640, 480))
    
    for i, frame in enumerate(frames):
        # Convert RGB to BGR for cv2
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
        
        if i % 50 == 0:
            print(f"  Written {i+1}/{len(frames)} frames")
    
    video_writer.release()
    print(f"✓ Video saved successfully to {video_filename}")
    
    # Print final height scanner stats
    # print("\nFinal height scanner analysis:")
    # try:
    #     final_height_map = env.get_height_map(rollout[-1].data)
    #     height_grid = final_height_map.reshape(5, 5)  # 5x5 grid
        
    #     print(f"Height map shape: {final_height_map.shape}")
    #     print(f"Height statistics:")
    #     print(f"  Min: {np.min(final_height_map):.3f}m")
    #     print(f"  Max: {np.max(final_height_map):.3f}m") 
    #     print(f"  Mean: {np.mean(final_height_map):.3f}m")
    #     print(f"  Std: {np.std(final_height_map):.3f}m")
        
    #     print(f"5x5 Height Grid (meters):")
    #     for row in height_grid:
    #         print("  " + " ".join(f"{val:6.3f}" for val in row))
            
    # except Exception as e:
    #     print(f"Could not analyze final height map: {e}")

if __name__ == "__main__":
    main()