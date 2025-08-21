#!/usr/bin/env python3
"""Test script for the rangefinder-based height scanner."""

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco.mjx._src.types import SensorType

def main():
    from mujoco_playground._src import registry
    env_name = 'Go1JoystickRoughTerrain'  # Use standard rough terrain
    
    env = registry.load(env_name)
    
    # Initialize the environment
    key = jax.random.PRNGKey(42)
    reset_key, step_key = jax.random.split(key)
    
    env_state = env.reset(reset_key)
    # Run a heigh scanner test.
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)

    site_pos = jp.asarray(env_state.data.site_xpos)
    sensor_types = jp.asarray(env.mjx_model.sensor_type)
    site_mat = jp.asarray(env_state.data.site_xmat)
    geom_xmat = jp.asarray(env_state.data.geom_xmat)
    geom_xpos = jp.asarray(env_state.data.geom_xpos)
    mj_data.geom_xmat = geom_xmat.reshape((-1, 9))
    mj_data.geom_xpos = geom_xpos
    height_readings = []
    # for i in range(sensor_types.shape[0]):
    #     if sensor_types[i] == SensorType.RANGEFINDER:
    #         objid = env.mjx_model.sensor_objid[i]
    #         site_bodyid = env.mjx_model.site_bodyid[objid]
            
    #         geom_filter = env.mjx_model.geom_bodyid != site_bodyid
    #         geom_filter &= True | (env.mjx_model.body_weldid[env.mjx_model.geom_bodyid] != 0)

    #         geom_filter_dyn = (env.mjx_model.geom_matid != -1) | (env.mjx_model.geom_rgba[:, 3] != 0)
    #         geom_filter_dyn &= (env.mjx_model.geom_matid == -1) | (env.mjx_model.mat_rgba[env.mjx_model.geom_matid, 3] != 0)
    #         site_id = env.mjx_model.sensor_objid[i]
            
    #         ray_pos = site_pos[site_id]
            
    #         ray_dir = site_mat[site_id].reshape((-1, 9))[:, np.array([2, 5, 8])]

    #         outputs = mujoco.mj_rayHfield(
    #             mj_model,
    #             mj_data,
    #             0, # Geom ID
    #             ray_pos,
    #             ray_dir[0],
    #         )
    #         height_readings.append(outputs)
    # height_readings = jp.array(height_readings)
    # # Make 5x5
    # mujoco_height_grid = height_readings.reshape(5, 5)
    # print("Height readings (5x5 grid):")
    # print(mujoco_height_grid)

    # mjx_height_grid = env.get_height_map_rangefinder(env_state.data).reshape(5, 5)
    # print("Height readings from mjx (5x5 grid):")
    # print(mjx_height_grid)
    
    # Test clearance computation by manually positioning the robot higher
    print("\n" + "="*60)
    print("TESTING CLEARANCE COMPUTATION")
    print("="*60)
    
    # Get current robot position and feet positions
    robot_pos = env_state.data.qpos[:3]  # x, y, z
    print(f"Original robot position: {robot_pos}")
    
    # Get foot positions and clearances at original height
    feet_site_id = np.array([env.mj_model.site(name).id for name in ["FL", "FR", "RL", "RR"]])
    original_foot_pos = env_state.data.site_xpos[feet_site_id]
    print(f"Original foot positions:\n{original_foot_pos}")
    
    # Debug foot body information
    print("Debugging foot body information:")
    for i, foot_name in enumerate(["FR", "FL", "RR", "RL"]):
        site_id = env.mj_model.site(foot_name).id
        body_id = env.mjx_model.site_bodyid[site_id]
        body_name = env.mj_model.body(body_id).name
        print(f"Foot {foot_name}: site_id={site_id}, body_id={body_id}, body_name={body_name}")
    
    # Also check what the current exclusion is
    root_body_id = env._mj_model.body("trunk").id
    print(f"Root body 'trunk' ID: {root_body_id}")
    
    # Test clearance calculation at original position
    original_clearances = env._get_terrain_height_below_feet(env.mj_model, env_state.data)
    print(f"Original clearances: {original_clearances}")
    
    # Manually move robot up by different heights to test clearance
    test_heights = [0.1, 0.2, 0.5, 1.0]
    
    for test_height in test_heights:
        print(f"\n--- Testing with robot lifted {test_height}m ---")
        
        # Create new qpos with robot lifted
        new_qpos = env_state.data.qpos.at[2].add(test_height)  # Lift z position
        
        # Create new data with modified position
        from mujoco import mjx
        test_data = env_state.data.replace(qpos=new_qpos)
        test_data = mjx.forward(env.mjx_model, test_data)
        
        # Get new foot positions
        new_foot_pos = test_data.site_xpos[feet_site_id]
        print(f"New foot positions (z-values): {new_foot_pos[:, 2]}")
        
        # Calculate clearances at new height
        clearances = env._get_terrain_height_below_feet(env.mj_model, test_data)
        print(f"Clearances: {clearances}")
        
        # Verify clearance makes sense (should be approximately foot_z - terrain_z)
        foot_z = new_foot_pos[:, 2]
        expected_clearances = foot_z - original_foot_pos[:, 2] + original_clearances
        print(f"Expected clearances: {expected_clearances}")
        print(f"Difference: {clearances - expected_clearances}")
        
        # Calculate clearance penalty using env's method
        feet_vel = test_data.sensordata[env._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2] 
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        
        min_clearance = 0.05
        desired_clearance = min_clearance + (env._config.reward_config.max_foot_height - min_clearance) * jp.tanh(vel_norm)
        clearance_error = jp.maximum(0, desired_clearance - clearances)
        clearance_penalty = jp.sum(clearance_error * vel_norm)
        
        print(f"Velocity norms: {vel_norm}")
        print(f"Desired clearances: {desired_clearance}")
        print(f"Clearance errors: {clearance_error}")  
        print(f"Total clearance penalty: {clearance_penalty}")
    
    print("\n" + "="*60)

    # step_jit = jax.jit(env.step)
    # # print(f"✓ Environment reset.")
    # # print("Warming up JIT compilation...")
    # for i in range(10):
    #     action = jp.zeros(env.action_size)  # Zero action (standing)
    #     env_state = step_jit(env_state, action)
        
    
    # # # Test a few simulation steps
    # # print("\nTesting simulation steps...")
    # import time
    # a = time.time()
    # try:
    #     for i in range(100):
    #         action = jp.zeros(env.action_size)  # Zero action (standing)
    #         env_state = step_jit(env_state, action)
    #         # height_map = env.get_height_map_rangefinder(env_state.data)
    #         # height_grid = height_map.reshape(5, 5)
    #         # print(height_grid)
    #         # print(f"Step {i+1}: Height range [{np.min(height_map):.3f}, {np.max(height_map):.3f}]m") 
    #         # print(f"Step {i+1}")#: Privileged state: {env_state.obs['privileged_state']}")
        
    #     print("✓ Simulation steps completed successfully!")
    
    # except Exception as e:
    #     print(f"✗ Failed during simulation steps: {e}")
    #     import traceback
    #     traceback.print_exc()
    # print(f"Total time for 100 steps: {time.time() - a:.2f} seconds")
    # print("\nRangefinder height scanner test completed!")

if __name__ == "__main__":
    main()