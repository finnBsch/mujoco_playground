# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Go1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import ray
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=1.0,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
              height_map=0.001,  # Noise on height map.
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              torso_height=-6.0,  # Adjust magnitude
              # Tracking.
              tracking_lin_vel=2.0,    ## 1.0
              tracking_ang_vel=0.5,    ## 0.5
              linear_orthogonal_velocity=0.0,
              world_direction=0.0,

              # Base reward.
              lin_vel_z=-0.5, # TODO   ## -0.5
              ang_vel_xy=-0.02,        ## -0.05
              orientation=-1.0,        ## -5.0
              # Other.
              dof_pos_limits=-0.01,    ## -1.0
              pose=1.0,  # 0.5         ## 0.5
              # Other.
              termination=-3.0,        ## -1.0
              stand_still=-0.1,        ## -1.0
              # Regularization.
              torques=-0.0004,        ## -0.0002
              action_rate=-0.01,       ## -0.01
              energy=-0.00003,         ## -0.001
              # Feet.
              feet_clearance=-0.1,     ## -2.0
              feet_height=-0.0,        ## -0.2
              feet_slip=-0.1,          ## -0.1
              feet_air_time=0.22,      ## 0.1
          ),
          tracking_sigma=0.25,
          max_foot_height=0.11,        ## 0.1
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[1.5, 0.8, 1.2],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      impl="jax",
      nconmax=4 * 8192,
      njmax=40,
  )


class Joystick(go1_base.Go1Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if task.startswith("rough"):
      config.nconmax = 100 * 8192
      config.njmax = 12 + 100 * 4
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    self._robot_body_ids = []
    
    # Method 1: Get all bodies that are part of the robot subtree
    # This assumes the robot is a connected kinematic chain starting from ROOT_BODY
    for body_id in range(self._mjx_model.nbody):
        # Check if this body is part of the robot by checking if it's in the subtree of ROOT_BODY
        current_body_id = body_id
        while current_body_id != 0:  # 0 is typically the world body
            parent_id = self._mjx_model.body_parentid[current_body_id]
            if parent_id == self._torso_body_id or current_body_id == self._torso_body_id:
                self._robot_body_ids.append(body_id)
                break
            current_body_id = parent_id

  def _reward_world_direction(self, info: dict, data: mjx.Data) -> jax.Array:
    """Encourage maintaining initial world direction when command was issued."""
    if "world_command" not in info:
        return jp.zeros(())
    
    global_vel = self.get_global_linvel(data)
    world_cmd = info["world_command"]
    
    # Only apply when there's a meaningful command
    cmd_magnitude = jp.linalg.norm(world_cmd[:2])
    
    def calculate_reward():
        world_vel_error = jp.sum(jp.square(world_cmd[:2] - global_vel[:2]))
        return jp.exp(-world_vel_error * 0.5)  # Softer than main tracking
    
    return jp.where(cmd_magnitude > 0.01, calculate_reward(), 0.0)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * 5.0
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )
    # Use our single-velocity sampling logic
    initial_cmd = jp.zeros(3)  # Start with zero command
    cmd = self.sample_command(key2, initial_cmd)
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    initial_yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # Convert local command to world coordinates
    cos_yaw = jp.cos(initial_yaw)
    sin_yaw = jp.sin(initial_yaw)
    world_cmd = jp.array([
        cos_yaw * cmd[0] - sin_yaw * cmd[1],
        sin_yaw * cmd[0] + cos_yaw * cmd[1],
        cmd[2]
    ])
    info = {
        "rng": rng,
        "command": cmd,
        "world_command": world_cmd,  # NEW: Initially same as local command
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    # state = self._reset_if_outside_bounds(state)

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]  # Absolute foot height
    terrain_distance = self._get_terrain_height_below_feet(self._mj_model, data)
    terrain_height = p_fz - terrain_distance  # Terrain z-coordinate below foot
    relative_swing_height = p_fz - terrain_height  # Height above terrain
    state.info["swing_peak"] = jp.where(
        contact,
        state.info["swing_peak"],  # Don't update if in contact
        jp.maximum(state.info["swing_peak"], relative_swing_height)
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["steps_until_next_cmd"] -= 1
    
    # NEW: Handle world_command updates when new commands are issued
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    new_local_cmd = self.sample_command(key1, state.info["command"])
    
    # Get current robot yaw to convert new local command to world coordinates
    trunk_quat = data.qpos[3:7]
    w, x, y, z = trunk_quat[0], trunk_quat[1], trunk_quat[2], trunk_quat[3]
    current_yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # Convert new local command to world coordinates
    cos_yaw = jp.cos(current_yaw)
    sin_yaw = jp.sin(current_yaw)
    new_world_cmd = jp.array([
        cos_yaw * new_local_cmd[0] - sin_yaw * new_local_cmd[1],
        sin_yaw * new_local_cmd[0] + cos_yaw * new_local_cmd[1],
        new_local_cmd[2]
    ])
    
    # Update both local and world commands when new command is issued
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        new_local_cmd,
        state.info["command"],
    )
    
    # NEW: Update world command only when new command is issued
    state.info["world_command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        new_world_cmd,
        state.info["world_command"]
    )
    
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    return fall_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    height_map = self._get_vertical_height_map(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_height_map = height_map
    # (
    #     height_map
    #     + (2 * jax.random.uniform(noise_rng, shape=height_map.shape) - 1)
    #     * self._config.noise_config.level
    #     * self._config.noise_config.scales.height_map)

    state = jp.hstack([
        noisy_linvel,  # 3, range [ ]
        noisy_gyro,  # 3, range [ ] 
        noisy_gravity,  # 3, range [ ] check if gravity or acc
        noisy_joint_angles - self._default_pose,  # 12. 
        noisy_joint_vel,  # 12. 
        #noisy_height_map,  # 100 range [0, 1.0] (10x10 grid)
        info["last_act"],  # 12 
        info["command"],  # 3
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)

    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        angvel,  # 3
        joint_angles - self._default_pose,  # 12
        joint_vel,  # 12
        #height_map,  # 100 (10x10 grid)
        data.actuator_force,  # 12
        info["last_contact"],  # 4
        feet_vel,  # 4*3
        info["feet_air_time"],  # 4
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "linear_orthogonal_velocity": self._reward_linear_orthogonal_velocity(
            info["command"], self.get_local_linvel(data)
        ),
        "world_direction": self._reward_world_direction(info, data),  # NEW
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, contact),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "torso_height": self._cost_torso_height(data),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
    }

  def _cost_torso_height(self, data: mjx.Data) -> jax.Array:
    """Penalize deviation from target torso height above terrain."""
    height_above_terrain = self._get_torso_terrain_height(data)
    target_height = 0.37  # Target height for Go1
    
    # Squared error from target height
    return jp.square(height_above_terrain - target_height)
  # Tracking rewards.

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_linear_orthogonal_velocity(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    """Reward for staying aligned with desired velocity direction.
    
    Implements r_lvo = exp(-3.0 * |v_o|²) where v_o is the orthogonal component
    of velocity relative to the desired direction.
    """
    # Get current velocity (xy only)
    v = local_vel[:2]
    
    # Get desired velocity magnitude
    cmd_magnitude = jp.linalg.norm(commands[:2])
    
    # Only apply reward if there's a meaningful command
    def calculate_reward():
        # Get desired velocity direction (normalized)
        v_des = commands[:2] / cmd_magnitude
        
        # Calculate orthogonal component: v_o = v - (v_des · v)v_des
        dot_product = jp.dot(v_des, v)
        v_parallel = dot_product * v_des
        v_o = v - v_parallel
        
        # Calculate reward: exp(- |v_o|²)
        v_o_squared_magnitude = jp.sum(jp.square(v_o))
        return jp.exp(- v_o_squared_magnitude)
    
    # Return reward only if command is meaningful, otherwise no reward (0.0)
    return jp.where(
        cmd_magnitude > 0.01,
        calculate_reward(),
        0.0
    )

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  # def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
  #   # Penalize non flat base orientation.
  #   return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array,
                         tolerance: float = 0.15,   ## Consider increasing this a bit
                         exp_scale: float = 7.0) -> jax.Array:
      """
      Penalize orientation outside a tolerance range with exponential growth.

      Args:
        torso_zaxis: Z-axis vector of torso orientation
        tolerance: Half-width of the dead zone (no penalty range) in radians
        exp_scale: Scale factor for exponential penalty growth

      Returns:
        Cost value (0 within tolerance, exponentially growing outside)
      """
      # Get pitch and roll components (first two elements)
      pitch_roll = torso_zaxis[:2]

      # Calculate absolute deviations
      abs_deviations = jp.abs(pitch_roll)

      # Calculate excess beyond tolerance (clipped to 0 if within tolerance)
      excess = jp.maximum(0.0, abs_deviations - tolerance)

      # Apply exponential penalty to excess
      penalties = jp.expm1(exp_scale * excess)  # expm1(x) = exp(x) - 1

      # Sum penalties for both pitch and roll
      return jp.sum(penalties)

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  # Other rewards.

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    # Stay close to the default pose.
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

  def _get_torso_terrain_height(self, data: mjx.Data) -> jax.Array:
    """Get torso height above terrain using multiple rays for robustness."""
    torso_pos = data.xpos[self._torso_body_id]
    
    # Cast rays in a small pattern around torso center
    offsets = jp.array([
        [0.0, 0.0],      # Center
        [0.1, 0.0],      # Forward
        [-0.1, 0.0],     # Back
        [0.0, 0.1],      # Right
        [0.0, -0.1],     # Left
    ])
    
    # Create ray start positions
    ray_starts = []
    for offset in offsets:
        start_pos = torso_pos + jp.array([offset[0], offset[1], 0.5])
        ray_starts.append(start_pos)
    
    ray_starts = jp.array(ray_starts)
    ray_dir = jp.array([0.0, 0.0, -1.0])
    
    # Batch ray cast
    distances, _ = ray.batch_ray(
        self._mj_model, data, ray_starts, ray_dir, (),
        True, bodyexclude=self._robot_body_ids
    )
    
    # Use minimum distance (highest terrain point under torso)
    min_distance = jp.min(distances)
    height_above_terrain = min_distance - 0.5
    
    return height_above_terrain

  def _get_terrain_height_below_feet(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        """Sample terrain height around each foot with a single batch ray cast."""
        foot_pos = data.site_xpos[self._feet_site_id]  # Shape: (4, 3)
        
        # Define sampling pattern (e.g., 5 points per foot: center + 4 around)
        # These are XY offsets from each foot
        sample_offsets = jp.array([
        # Center point
        [0.0, 0.0],

        # Close cardinal directions (0.05 units)
        [0.05, 0.0],      # Forward
        [-0.05, 0.0],     # Back  
        [0.0, 0.05],      # Right
        [0.0, -0.05],     # Left

        # Close diagonal directions (0.05 units)
        [0.05, 0.05],     # Forward-right
        [0.05, -0.05],    # Forward-left
        [-0.05, 0.05],    # Back-right
        [-0.05, -0.05],   # Back-left

        # # # Far cardinal directions (0.1 units)
        # [0.2, 0.0],       # Far forward
        # [-0.2, 0.0],      # Far back
        # [0.0, 0.2],       # Far right
        # [0.0, -0.2],      # Far left

        # # Far diagonal directions (0.2 units)
        # [0.2, 0.2],       # Far forward-right
        # [0.2, -0.2],      # Far forward-left
        # [-0.2, 0.2],      # Far back-right
        # [-0.2, -0.2],     # Far back-left
        ])  # Shape: (17, 2)
      
        # Create all sample positions for all feet at once
        # Broadcast foot positions with offsets
        num_samples = sample_offsets.shape[0]
        
        # Expand dimensions for broadcasting
        foot_pos_expanded = foot_pos[:, None, :]  # (4, 1, 3)
        offsets_3d = jp.pad(sample_offsets[None, :, :], ((0,0), (0,0), (0,1)))  # (1, 5, 3)
        # print(foot_pos)
        # All sample positions: (4 feet * 5 samples = 20 total)
        sample_positions = (foot_pos_expanded + offsets_3d).reshape(-1, 3)  # (20, 3)
        # print(sample_positions)

        # Start rays 0.5 units above the sample positions
        ray_start_offset = jp.array([0.0, 0.0, 0.5])
        elevated_sample_positions = sample_positions + ray_start_offset  # (20, 3)
        
        # Single ray direction for all samples (straight down)
        ray_dir = jp.array([0.0, 0.0, -1.0])
        
        # Single batch ray cast for all samples
        distances, geom_ids = ray.batch_ray(
            model, data, elevated_sample_positions, ray_dir, (), 
            True, bodyexclude=self._robot_body_ids
        )  # Shape: (20,)
        # print(geom_ids.min())
        # print(self._robot_body_ids)
        # Reshape to (4 feet, 5 samples) and take max height around each foot
        distances_per_foot = distances.reshape(4, num_samples)
        
        # Get the minimum distance (highest terrain) around each foot
        # Note: smaller distance = higher terrain when casting downward
        min_distances = jp.min(distances_per_foot, axis=1)  # Shape: (4,)
        
        # Subtract the 0.5 offset to get terrain height relative to original foot position
        # Positive values = terrain below foot, Negative values = terrain above foot
        terrain_heights = min_distances - 0.5
        
        return terrain_heights

  def _get_vertical_height_map(self, data: mjx.Data) -> jax.Array:
        """Get height map using vertical ray casting independent of robot tilt.
        
        Creates a 10x10 grid of vertical rays centered on the robot's position,
        casting downward in world coordinates. Grid rotates with robot's yaw
        but rays remain vertical regardless of robot pitch/roll.
        """
        # Get robot's world position and orientation from root body (freejoint)
        trunk_pos = data.qpos[0:3]  # (3,) - root body position
        trunk_quat = data.qpos[3:7]  # (4,) - root body quaternion [w, x, y, z]
        
        # Extract yaw angle from quaternion
        # For quaternion [w, x, y, z], yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        w, x, y, z = trunk_quat[0], trunk_quat[1], trunk_quat[2], trunk_quat[3]
        yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # Create rotation matrix for yaw only (around Z-axis)
        cos_yaw = jp.cos(yaw)
        sin_yaw = jp.sin(yaw)
        rotation_matrix = jp.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw,  cos_yaw]
        ])
        
        # Create 10x10 grid of sampling positions relative to robot center
        # Grid covers 1m x 1m area (±0.5m in each direction) with ~0.111m spacing
        grid_positions = []
        for i in range(10):
            for j in range(10):
                # Convert grid indices to robot-relative offsets
                x_robot = -0.5 + i * (1.0 / 9)  # -0.5 to +0.5 (forward/back in robot frame)
                y_robot = -0.5 + j * (1.0 / 9)  # -0.5 to +0.5 (left/right in robot frame)
                
                # Rotate the offset by robot's yaw to get world coordinates
                robot_offset = jp.array([x_robot, y_robot])
                world_offset = rotation_matrix @ robot_offset
                
                # Create world position by adding rotated offset to robot position
                world_pos = jp.array([
                    trunk_pos[0] + world_offset[0],
                    trunk_pos[1] + world_offset[1], 
                    trunk_pos[2] + 0.5  # Start ray 0.5m above robot
                ])
                grid_positions.append(world_pos)
        
        # Convert to array: (100, 3)
        ray_start_positions = jp.array(grid_positions)
        
        # Debug: Print first few ray positions for comparison
        # print(f"Robot pos: {trunk_pos}")
        # print(f"Robot yaw: {yaw}")
        # print(f"First 5 ray start positions:\n{ray_start_positions[:5]}")
        
        # Vertical downward ray direction (world coordinates - always straight down)
        ray_dir = jp.array([0.0, 0.0, -1.0])
        
        # Cast all rays at once
        distances, _ = ray.batch_ray(
            self._mj_model, data, ray_start_positions, ray_dir, (),
            True, bodyexclude=self._robot_body_ids
        )  # Shape: (100,)
        
        # Convert distances to heights relative to robot
        # Subtract 0.5 to account for ray start offset
        # Positive values = terrain below robot, Negative = terrain above robot
        relative_heights = distances - 0.5
        
        return relative_heights  # Shape: (100,)

  def _cost_feet_clearance(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
        """Penalize insufficient clearance during swing phase only."""
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        
        # Get terrain clearance
        clearance = self._get_terrain_height_below_feet(self._mj_model, data)
        
        # Minimum safe clearance
        min_safe_clearance = self._config.reward_config.max_foot_height
        
        # Only penalize insufficient clearance, and only during swing phase
        insufficient_clearance = jp.maximum(0, min_safe_clearance - clearance)

        return jp.sum(insufficient_clearance * vel_norm * (~contact))

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
      """Penalize swing height error."""
      cmd_norm = jp.linalg.norm(info["command"])
      error = swing_peak / self._config.reward_config.max_foot_height - 1.0
      return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum(jp.exp(-jp.square(air_time - 0.8)) * first_contact)
    rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
    return rew_air_time

  # Perturbation and command sampling.

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
      force = (
          u_t  # (unitless)
          * self._torso_mass  # kg
          * state.info["pert_mag"]  # m/s
          / state.info["pert_duration_seconds"]  # 1/s
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )

  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, choice_rng, y_rng, w_rng, z_rng = jax.random.split(rng, 5)
    
    # Choose which type of velocity to sample (0: forward/back, 1: left/right, 2: angular)
    velocity_type = jax.random.choice(choice_rng, 3)
    
    # Sample values for all dimensions
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    
    # Create mask to only update one velocity type
    velocity_mask = jp.zeros(3)
    velocity_mask = velocity_mask.at[velocity_type].set(1.0)
    
    # Apply mask so only one velocity type gets updated, others stay at current value
    x_kp1 = x_k - w_k * velocity_mask * (x_k - y_k * z_k)
    return x_kp1