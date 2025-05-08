""" This code describes a training environment for the NVIDIA Isaac Gym 
simulator.

"""
import csv
import math
import numpy as np
import os
import torch
import random
from isaacgym import gymutil, gymtorch, gymapi

# from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float, tensor_clamp, torch_random_dir_2
from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *



class BallBalance(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.obs_history = []
        self.ball_pos_history = []  
        self.tray_pos_history = [] 
        self.REAL_BALL_pos_history = []
        self.cfg = cfg
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.action_speed_scale = self.cfg["env"]["actionSpeedScale"]
        self.debug_viz = True #self.cfg["env"]["enableDebugVis"]
        self.smoothness_weight = 0.5 
        self.ball_heigh = 0.3
        sensors_per_env = 1
        actors_per_env = 2
        dofs_per_env = 7
        bodies_per_env = 1 + 1  #7+1

        # Observations:
        # 0:3 - activated DOF positions
        # 3:6 - activated DOF velocities
        # 6:9 - ball position
        # 9:12 - ball linear velocity
        # 12:15 - sensor force (same for each sensor)
        # 15:18 - sensor torque 1
        # 18:21 - sensor torque 2
        # 21:24 - sensor torque 3
        self.cfg["env"]["numObservations"] = 18

        # Actions: target velocities for the 3 actuated DOFs
        self.cfg["env"]["numActions"] = 7

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, actors_per_env, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)
        vec_rigid_body_tensor = gymtorch.wrap_tensor(self.rigid_body_tensor).view(self.num_envs, self.num_body + 1, 13)
        # vec_sensor_tensor = gymtorch.wrap_tensor(self.sensor_tensor).view(self.num_envs, sensors_per_env, 6)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.custom_obs = [
            -1.35115399e-01, 9.43089922e-02, 1.27942127e-01, -1.27216037e-01,
            1.57000000e+00, 0.00000000e+00, -3.00000000e-01, 3.02732378e-01,
            -8.44771467e-01, -6.41350331e-01, -4.89879824e-01, -4.62962963e-03,
            -1.53614458e-01, -4.17029436e-04, -1.15410415e-02, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00
        ]

        self.root_states = self.root_states
        self.tray_positions = vec_rigid_body_tensor[..., self.bbot_tray_idx, 0:3]
        self.tray_orientations = vec_rigid_body_tensor[..., self.bbot_tray_idx, 3:7]
        self.ball_positions = self.root_states[..., 1, 0:3]
        self.ball_orientations = self.root_states[..., 1, 3:7]
        self.ball_linvels = self.root_states[..., 1, 7:10]
        self.ball_angvels = self.root_states[..., 1, 10:13]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]
        # self.dof_positions [...,0:6]  = self.custom_obs[0:6]
        # self.sensor_forces = vec_sensor_tensor[..., 0:3]
        # self.sensor_torques = vec_sensor_tensor[..., 3:6]

        self.initial_tray_positions = self.tray_positions.clone()
        self.initial_tray_orientations = self.tray_orientations.clone()
        self.initial_dof_states = self.dof_states.clone()
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_positions = self.dof_positions.clone()
        
        self.dof_position_targets = torch.zeros((self.num_envs, dofs_per_env), dtype=torch.float32, device=self.device, requires_grad=False)
        # print (self.dof_position_targets )
        self.all_actor_indices = torch.arange(actors_per_env * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, actors_per_env)
        self.all_bbot_indices = actors_per_env * torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        buff_size = 10
        self.action_buffer = torch.zeros(self.num_envs,buff_size,self.cfg["env"]["numActions"],device=self.device)
        

        # vis
        self.axes_geom = gymutil.AxesGeometry(0.5)

        self.reset_idx(torch.arange(0,self.num_envs,device=self.device))

    def create_sim(self):
            

            
            
            
            self.dt = self.sim_params.dt
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
            self.sim_params.gravity.x = 0
            self.sim_params.gravity.y = 0
            self.sim_params.gravity.z = -9.81

            self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

            
            self._create_ground_plane()
            self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
            if self.randomize:
                self.apply_randomizations(self.randomization_params)



    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
     
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/upper_thormang_copy.urdf"
        # Load asset
        
        
        
        bbot_options = gymapi.AssetOptions()
        bbot_options.fix_base_link = False
        # bbot_options.slices_per_cylinder = 40
        
        bbot_options.flip_visual_attachments = False
        bbot_options.fix_base_link = True
        bbot_options.collapse_fixed_joints = False
        bbot_options.disable_gravity = False
        bbot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, bbot_options)
        

        # printed view of asset built
        # self.gym.debug_print_asset(bbot_asset)

        self.num_bbot_dofs = self.gym.get_asset_dof_count(bbot_asset)

        bbot_dof_props = self.gym.get_asset_dof_properties(bbot_asset)
        self.bbot_dof_lower_limits = []
        self.bbot_dof_upper_limits = []
        for i in range(self.num_bbot_dofs):
            self.bbot_dof_lower_limits.append(bbot_dof_props['lower'][i])
            self.bbot_dof_upper_limits.append(bbot_dof_props['upper'][i])

        self.bbot_dof_lower_limits = to_torch(self.bbot_dof_lower_limits, device=self.device)
        self.bbot_dof_upper_limits = to_torch(self.bbot_dof_upper_limits, device=self.device)

        bbot_pose = gymapi.Transform()
        bbot_pose.p.z = 1.0
        # bbot_pose.p.x = 100.0


        
        self.bbot_tray_idx = self.gym.find_asset_rigid_body_index(bbot_asset, "tray")
        self.num_body = self.gym.get_asset_rigid_body_count(bbot_asset)
        # print(self.bbot_tray_idx)
    

        # create ball asset
        self.ball_radius = 0.019
        ball_options = gymapi.AssetOptions()
        ball_options.density = 84
        # ball_options.vhacd_params.resolution = 300000
        # ball_options.vhacd_params.max_convex_hulls = 10
        # ball_options.vhacd_params.max_num_vertices_per_ch = 64
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)

        self.envs = []
        self.bbot_handles = []
        self.obj_handles = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            
            # print(self.gym.get_asset_dof_names(bbot_asset))
            bbot_handle = self.gym.create_actor(env_ptr, bbot_asset, bbot_pose, "bbot", i, 0, 0)
            
            actuated_dofs = np.array([[0,1,2,3]])
            hand_dof = np.array([4,5,6])
            # fixed_dof =  np.arange(21)
            # fixed_dof = np.delete(fixed_dof, [11,12,13,14,15,16,17])
            # free_dofs = np.array([0, 2, 4])
            # actuated_dofs = self.gym.get_asset_actuator_count(bbot_asset)
            # print(self.gym.get_actor_rigid_body_names(env_ptr, bbot_handle))
            # free_dofs =self.gym.get_asset_dof_count(bbot_asset)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, bbot_handle)
            
            dof_props['driveMode'][actuated_dofs] = gymapi.DOF_MODE_POS
            dof_props['damping'][actuated_dofs ] =8200.0#100#8200.0
            dof_props['stiffness'][actuated_dofs ] =59500.0#2500#59500.0
            dof_props['effort'][actuated_dofs ] = 100#10.7 #4407
            dof_props['velocity'][actuated_dofs ] = 3.03 #5.24 #3.03


            dof_props['driveMode'][hand_dof] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][hand_dof] = 2500
            dof_props['damping'][hand_dof] = 100

            dof_props['effort'][hand_dof] = 10
            dof_props['velocity'][hand_dof] = 10
            # pos_targets = lower_limits + ranges * np.random.random(num_dofs).astype('f')
            # self.gym.set_actor_dof_position_targets(env_ptr, bbot_handle, pos_targets)
           
            self.gym.set_actor_dof_properties(env_ptr, bbot_handle, dof_props)

            ball_pose = gymapi.Transform()
            ball_pose.p.x = 10.0
            ball_pose.p.y = 10.0
            ball_pose.p.z = 10.0
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, 0, 0)
            self.obj_handles.append(ball_handle)


            # ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr,  ball_handle)

          
            # for s in ball_shape_props:
            #     s.friction = 0  
            # self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, ball_shape_props)

          
            # # pretty colors
            # self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.99, 0.66, 0.25))
            # self.gym.set_rigid_body_color(env_ptr, bbot_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.48, 0.65, 0.8))
            # for j in range(1, 7):
            #     self.gym.set_rigid_body_color(env_ptr, bbot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.2, 0.3))

            self.envs.append(env_ptr)
            self.bbot_handles.append(bbot_handle)
            
        

    def compute_observations(self):
        #print("~!~!~!~! Computing obs")

        actuated_dof_indices = torch.tensor([0,1,2,3,4,5,6], device=self.device)
        v_dof_indices = torch.tensor([0,1,2,3], device=self.device)
        #print(self.dof_states[:, actuated_dof_indices, :])
        


        self.obs_buf[..., 0:4] = self.dof_positions[...,[0,1,2,3]]
        # print(self.obs_buf[10, 0:7])
        self.obs_buf[..., 4:7] = self.dof_positions[...,[4,5,6]]
        # print("action_after_scale",self.obs_buf[10,0:7])
        # print("DOF Positions (0:7):", self.dof_positions[..., 4])

        self.obs_buf[..., 7:11] = self.dof_velocities[..., v_dof_indices]
        noise = 0.007
        self.obs_buf[..., 0:4] += torch.empty(self.obs_buf[..., 0:4].shape, device=self.obs_buf.device).uniform_(-noise, noise)
        self.obs_buf[..., 7:11] += torch.empty(self.obs_buf[..., 7:11].shape, device=self.obs_buf.device).uniform_(-noise, noise)
        # print(self.obs_buf[10, 7:11])
        # print("DOF Velocities (7:14):", self.dof_velocities[..., actuated_dof_indices])
        ball_to_corner_vector = self.ball_positions - self.tray_positions
        relative_pos_local = quat_rotate(quat_conjugate(self.tray_orientations), ball_to_corner_vector)
        # print(relative_pos_local[10])
        self.obs_buf[..., 11:13] = - torch.abs(relative_pos_local[...,[1,2]])
        # print("ball_position",self.obs_buf[10, 11:13])

        noise_range = 0.05  
        # self.obs_buf[..., 11:13] =torch.round((self.obs_buf[:,11:13]*100))/100
        # print("ball_position",self.obs_buf[10, 11:13])
        # print("apply,rotate",relative_pos_local[10,:],rotate[10,:])
        # print(self.obs_buf[15, 14:16])
        # if(torch.any(self.obs_buf[15, 14:16]<0)): #and torch.any(self.obs_buf[..., 14:16]>0.25) ):
        #     print("oups")
        #     input()
        # # print({self.ball_positions[..., [0, 1]]},{self.tray_positions[..., [0, 1]]})
        # print("Ball positions (X, Y) - Tray positions (X, Y):", self.ball_positions, "-", self.tray_positions)
        
        ball_velocity = quat_rotate(quat_conjugate(self.tray_orientations), self.ball_linvels)
        self.obs_buf[..., 13:15] = ball_velocity[..., [1, 2]]
        # print('vel',self.obs_buf[10, 13:15])
        # print("showspeed",ball_velocity[10] )
        self.obs_buf[..., 11:15] += torch.empty(self.obs_buf[..., 11:15].shape, device=self.obs_buf.device).uniform_(-noise_range, noise_range)
        corner_to_center_vector = torch.zeros_like(self.tray_positions)
        corner_to_center_vector[:, 0] = -0.002
        corner_to_center_vector[:, 1] = -0.25/2
        corner_to_center_vector[:, 2] = -0.25/2
        self.obs_buf[..., 15:18] = 0.0

        # print(tray_orientations[0])
        # print(corner_to_center_vector[0])
        # print()
        
        rotated_offset = quat_rotate(self.tray_orientations, corner_to_center_vector)
        center_pos = self.tray_positions + rotated_offset
        
        ball_pos = self.ball_positions
        to_target = ball_pos - center_pos
        # print("obs",self.obs_buf[1])
        # print("dof_state",self.dof_positions[10])
        # print("position",self.obs_buf[10, 11:13])
        # print("speed",self.obs_buf[10, 13:15])

        return self.obs_buf
    


    def compute_reward(self):

        self.rew_buf[:], self.reset_buf[:] = compute_bbot_reward(
        self.tray_positions,
        self.tray_orientations,
        self.ball_positions,
        self.ball_linvels,
        # self.dof_positions,
        self.reset_buf, 
        self.progress_buf, 
        self.max_episode_length,
        self.action_buffer,  # 傳遞當前的動作
        self.smoothness_weight ,
        # self.ball_heigh
        

    )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
       
        # num_resets = len(env_ids)
        self.dof_states[env_ids] = self.initial_dof_states[env_ids]

        # reset bbot and ball root states
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.action_buffer[env_ids] = torch.zeros(len(env_ids),10,self.cfg["env"]["numActions"],device=self.device)
        positions = torch_rand_float(-0.01, 0.01, (len(env_ids), self.num_bbot_dofs), device=self.device)
        
        # velocities = torch_rand_float(-0.001, 0.001, (len(env_ids), self.num_bbot_dofs), device=self.device)
        velocities = torch.zeros((len(env_ids), self.num_bbot_dofs), device=self.device)

        # self.initial_dof_positions[env_ids, 4] = 1.57

        self.dof_positions[env_ids] = tensor_clamp(self.initial_dof_positions[env_ids] , self.bbot_dof_lower_limits, self.bbot_dof_upper_limits)
        self.dof_velocities[env_ids] = velocities 
        
        
        # print(self.dof_position_targets)
        # self.root_states[env_ids] = self.initial_root_states[env_ids].clone()

        corner_to_center_vector = torch.zeros_like(self.tray_positions)
        corner_to_center_vector[:, 0] = -0.002
        corner_to_center_vector[:, 1] = -0.25/2
        corner_to_center_vector[:, 2] = -0.25/2
        

        # print(self.tray_orientations[0])
        # print(corner_to_center_vector[0])
        # print()

        rotated_offset = quat_rotate(self.initial_tray_orientations, corner_to_center_vector)
        center_pos = self.initial_tray_positions + rotated_offset

        # self.ball_positions[env_ids, 0] = self.tray_positions[env_ids,0]
        # self.ball_positions[env_ids, 0] = self.tray_positions[env_ids,2]
        # self.ball_positions[env_ids, 0] = self.tray_positions[env_ids,1]
        self.ball_positions[env_ids, 0] = center_pos[env_ids,0] + random.uniform(-0.08, 0.08)
        self.ball_positions[env_ids, 2] = center_pos[env_ids,2] + 0.02
        self.ball_positions[env_ids, 1] = center_pos[env_ids,1] + random.uniform(-0.08, 0.08)
        # print(self.ball_positions[env_ids, 0],self.ball_positions[env_ids, 1],{self.ball_positions[env_ids, 2],})

        # print(self.ball_positions[0])
        # print(center_pos[0])



        self.ball_orientations[env_ids, 0:3] = 0
        self.ball_orientations[env_ids, 3] = 1
        self.ball_linvels[env_ids, 0] = 0
        self.ball_linvels[env_ids, 2] = 0
        self.ball_linvels[env_ids, 1] = 0
        self.ball_angvels[env_ids] = 0


        # reset root state for bbots and balls in selected envs
        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

        # reset DOF states for bbots in selected envs
        bbot_indices = self.all_bbot_indices[env_ids].flatten()
        
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(bbot_indices), len(bbot_indices))   
     
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_states[...,0].contiguous()),
                                                        gymtorch.unwrap_tensor(bbot_indices), 
                                                        len(bbot_indices)
                                                        )

        # self.gym.set_dof_state_tensor_indexed(
        #                                         self.sim,
        #                                         gymtorch.unwrap_tensor(self.dof_positions),
        #                                         gymtorch.unwrap_tensor(bbot_indices),
        #                                         len(bbot_indices)
        #                                      )

        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0 
    # def reset_idx(self, env_ids):
    #     if self.randomize:
    #         self.apply_randomizations(self.randomization_params)
       
    #     # num_resets = len(env_ids)
    #     self.dof_states[env_ids] = self.initial_dof_states[env_ids]

    #     # reset bbot and ball root states
    #     self.root_states[env_ids] = self.initial_root_states[env_ids]
    #     self.action_buffer[env_ids] = torch.zeros(len(env_ids),10,self.cfg["env"]["numActions"],device=self.device)
    #     positions = torch_rand_float(-0.01, 0.01, (len(env_ids), self.num_bbot_dofs), device=self.device)
        
    #     # velocities = torch_rand_float(-0.001, 0.001, (len(env_ids), self.num_bbot_dofs), device=self.device)
    #     velocities = torch.zeros((len(env_ids), self.num_bbot_dofs), device=self.device)

    #     # self.initial_dof_positions[env_ids, 4] = 1.57

    #     self.dof_positions[env_ids] = tensor_clamp(self.initial_dof_positions[env_ids] , self.bbot_dof_lower_limits, self.bbot_dof_upper_limits)
    #     self.dof_velocities[env_ids] = velocities 
        
        
    #     # print(self.dof_position_targets)
    #     # self.root_states[env_ids] = self.initial_root_states[env_ids].clone()

    #     corner_to_center_vector = torch.zeros_like(self.tray_positions)
    #     corner_to_center_vector[:, 0] = -0.002
    #     corner_to_center_vector[:, 1] = -0.25/2
    #     corner_to_center_vector[:, 2] = -0.25/2
        
        
    #     # print(self.tray_orientations[0])
    #     # print(corner_to_center_vector[0])
    #     # print()

    #     rotated_offset = quat_rotate(self.initial_tray_orientations, corner_to_center_vector)
    #     center_pos = self.initial_tray_positions + rotated_offset

    #     # self.ball_positions[env_ids, 0] = self.tray_positions[env_ids,0]
    #     # self.ball_positions[env_ids, 0] = self.tray_positions[env_ids,2]
    #     # self.ball_positions[env_ids, 0] = self.tray_positions[env_ids,1]
    #     self.ball_positions[env_ids, 0] = self.custom_obs[env_ids,12]
    #     self.ball_positions[env_ids, 2] = center_pos[env_ids,2] + 0.02
    #     self.ball_positions[env_ids, 1] = self.custom_obs[env_ids,13]

    #     # print(self.ball_positions[env_ids, 0],self.ball_positions[env_ids, 1],{self.ball_positions[env_ids, 2],})

    #     # print(self.ball_positions[0])
    #     # print(center_pos[0])



    #     self.ball_orientations[env_ids, 0:3] = 0
    #     self.ball_orientations[env_ids, 3] = 1
    #     self.ball_linvels[env_ids, 0] = self.custom_obs[env_ids,14]
    #     self.ball_linvels[env_ids, 2] = 0
    #     self.ball_linvels[env_ids, 1] = self.custom_obs[env_ids,15]
    #     self.ball_angvels[env_ids] = 0
        

    #     # reset root state for bbots and balls in selected envs
    #     actor_indices = self.all_actor_indices[env_ids].flatten()
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

    #     # reset DOF states for bbots in selected envs
    #     bbot_indices = self.all_bbot_indices[env_ids].flatten()
        
    #     self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(bbot_indices), len(bbot_indices))   
     
    #     self.gym.set_dof_position_target_tensor_indexed(self.sim,
    #                                                     gymtorch.unwrap_tensor(self.dof_states[...,0].contiguous()),
    #                                                     gymtorch.unwrap_tensor(bbot_indices), 
    #                                                     len(bbot_indices)
    #                                                     )

    #     # self.gym.set_dof_state_tensor_indexed(
    #     #                                         self.sim,
    #     #                                         gymtorch.unwrap_tensor(self.dof_positions),
    #     #                                         gymtorch.unwrap_tensor(bbot_indices),
    #     #                                         len(bbot_indices)
    #     #                                      )

    #     # self.gym.set_dof_state_tensor_indexed(self.sim,
    #     #                                       gymtorch.unwrap_tensor(self.dof_state),
    #     #                                       gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
    #     self.reset_buf[env_ids] = 0
    #     self.progress_buf[env_ids] = 0     

    def pre_physics_step(self, _actions):
        # _actions = _actions *0.0
        # _actions = torch.tensor([-0.4878, -1.0000, -1.0000,  0.1501,  0.3681, -1.0000,  0.1755], device='cuda:0')
        # _actions[:,4] =  _actions[:,4]+1.57
        self.action_buffer[:,:-1] = self.action_buffer[:,1:].clone()
        # print('action',_actions[1])
        self.action_buffer[:,-1]  = _actions 
        # print("action_buffer.shape:", self.action_buffer.shape)


        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        position_tensor = self.dof_positions.clone()
        self.actions = _actions.to(self.device)
        # print("action",_actions[10,...])
        # print(self.actions[10,...] )
        # with open('/home/peter/isaacgym/python/examples/IsaacGymEnvs/isaacgymenvs/tasks/actions.csv', mode='r') as file:#real to sim
        #     reader = csv.reader(file)
            
        #     next(reader)  
        #     for row in reader:
        # 
        #         static_actions = [float(value) for value in row] 
        #         print("Loaded static_obs:", static_actions)

        # if len(static_actions) != 7:
        #     raise ValueError(f"Observation length is {len(static_actions)}, expected 18.")
        # self.actions  = torch.tensor([static_actions]*self.num_envs, dtype=torch.float32, device=self.rl_device)  #real to sim

        # excluded_indices = torch.LongTensor(range(26, 33))
        # mapped_indices = torch.LongTensor([i for i in range(49) if i not in excluded_indices])
        # result = mapped_indices[torch.LongTensor([])]
        
        actuated_indices = torch.LongTensor([0,1,2,3,4,5,6])
        
        # update position targets from actions
        # print(self.dof_position_targets[..., actuated_indices])
        # print(self.dt * self.action_speed_scale * actions)
        # from math import pi
        # target_angle = -pi / 2
        # self.dof_position_targets[:, 1] = target_angle

        # # Print updated dof position targets for debugging
        # print("Updated dof position targets with second joint set to -π/2:", self.dof_position_targets)

        # Clamp the values to be within the joint limits
        

        #self.dof_position_targets[..., actuated_indices] += self.dt * self.action_speed_scale * self.actions
        
        # print("after scale",self.dof_position_targets[10, actuated_indices])
       
        
        # # self.dof_position_targets[..., result] = 0
        # self.dof_position_targets[:] = tensor_clamp(self.dof_position_targets, self.bbot_dof_lower_limits, self.bbot_dof_upper_limits)
        # print("after scale and clamp",self.dof_position_targets[10, actuated_indices])
        # reset position targets for reset envs
        
        self.dof_position_targets[..., actuated_indices] =  (self.actions+1) * 0.5 * (self.bbot_dof_upper_limits-self.bbot_dof_lower_limits) + self.bbot_dof_lower_limits

        # self.dof_position_targets[result] = 0.0
        # if(self.progress_buf[0] == 1):
        #     self.dof_position_targets[..., actuated_indices] =  (self.actions+1)*0.5*(self.bbot_dof_upper_limits-self.bbot_dof_lower_limits)+self.bbot_dof_lower_limits
        # print("after scale",self.dof_position_targets[10, actuated_indices])
        self.dof_position_targets[reset_env_ids] = 0
        # print("DOF Positions:", self.dof_positions[10])
        # print("DOF Targets:", self.dof_position_targets[1])
        # print("Observations:", self.obs_buf[10])
        # print("Simulating step...")
        # self.gym.simulate(self.sim)
        # self.gym.fetch_results(self.sim, True)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))
        
        
        
    def post_physics_step(self):

        self.progress_buf += 1
        self.randomize_buf += 1
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # if self.progress_buf.float().mean() < 0.5 * self.max_episode_length:
        #     self.smoothness_weight = 0.1  
        # else:
        #     self.smoothness_weight = 0.5  
        progress_ratio = (self.progress_buf.float().mean() / self.max_episode_length).clamp(0, 1)
        self.ball_heigh =  progress_ratio * 1.8
        self.smoothness_weight = 1.5 + progress_ratio * (3.8 - 1.5)

        self.compute_observations()
        self.compute_reward()
        # print(self.tray_positions)
        # vis
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            points = []
            colors = []
            corner_to_center_vector = torch.zeros_like(self.tray_positions)
            corner_to_center_vector[:, 0] = -0.002
            corner_to_center_vector[:, 1] = -0.25/2
            corner_to_center_vector[:, 2] = -0.25/2
            # print("tray_orientaion",self.tray_orientations[10])
            # print("ballposition",self.ball_positions[10])
            # print("tray_position",self.tray_positions[10])
            rotated_offset = quat_rotate(self.tray_orientations, corner_to_center_vector)
            center_pos = self.tray_positions + rotated_offset
            
            ball_to_corner_vector = self.ball_positions - self.tray_positions
            rotated_offset = quat_rotate(quat_conjugate(self.tray_orientations), ball_to_corner_vector)
            relative_pos_global = quat_rotate(self.tray_orientations, rotated_offset) + self.tray_positions
           
            

            for i in range(self.num_envs):

                env = self.envs[i]
                bbot_handle = self.bbot_handles[i]
                body_handles = []
                body_handles.append(self.gym.find_actor_rigid_body_handle(env, bbot_handle, "tray"))
                # body_handles.append(self.gym.find_actor_rigid_body_handle(env, bbot_handle, "upper_leg1"))
                # body_handles.append(self.gym.find_actor_rigid_body_handle(env, bbot_handle, "upper_leg2"))

                

                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*center_pos[i])
                pose.r = gymapi.Quat(*self.tray_orientations[i])

                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose)
                # pose2 = gymapi.Transform()
                # pose2.p=gymapi.Vec3(*center_pos[i])
                # pose2.r = gymapi.Quat(*revers[i])
                # print("pose2",pose2.p)
                # gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose2)
                
                pose_ball = gymapi.Transform()
                pose_ball.p = gymapi.Vec3(*relative_pos_global[i])
                pose_ball.r = gymapi.Quat(*self.tray_orientations[i])
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, pose_ball)

                corner = gymapi.Transform(p=gymapi.Vec3(*self.tray_positions[i]), r=gymapi.Quat(*self.tray_orientations[i]))
                
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, corner)

                # for lhandle in body_handles:
                #     lpose = self.gym.get_rigid_transform(env, lhandle)
                #     gymutil.draw_lines(self.axes_geom, self.gym, sself.gym, env, lpose)
                


#####################################################################
###=========================jit functions=========================###
#####################################################################


#@torch.jit.script
def compute_bbot_reward(tray_positions, tray_orientations, ball_positions, ball_velocities , reset_buf, progress_buf, max_episode_length, action_buffer, smoothness_weight ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    # calculating the norm for ball distance to desired height above the ground plane (i.e. 0.7)

    action_diff = torch.diff(action_buffer, dim=1)
    #[nenv, 7]

    action_diff_penalty = torch.mean(torch.abs(action_diff),dim=(1,2))/9.0

    #action_diff_penalty = torch.cat([action_diff_penalty, torch.zeros(1, device=action_diff_penalty.device)])
    smoothness_penalty =  action_diff_penalty*smoothness_weight

    corner_to_center_vector = torch.zeros_like(tray_positions)
    corner_to_center_vector[:, 0] = -0.002
    corner_to_center_vector[:, 1] = -0.25/2
    corner_to_center_vector[:, 2] = -0.25/2
    
    # print(tray_orientations[0])
    # print(corner_to_center_vector[0])
    # print()
    
    # rotated_offset = quat_rotate(tray_orientations, corner_to_center_vector)
    # center_pos = tray_positions + rotated_offset
    # corner_to_center_vector = quat_rotate(quat_conjugate(tray_orientations), rotated_offset)

    ball_pos = ball_positions
    
    ball_to_corner_vector = ball_positions - tray_positions
    relative_ball_pos = quat_rotate(quat_conjugate(tray_orientations), ball_to_corner_vector)
    
    relative_center_pose = torch.tensor([0.019/2, 0.25/2,0.25/2], device='cuda:0')

    to_target = torch.abs(relative_ball_pos) - torch.abs(relative_center_pose)
    # print("reward_relative_ball_pos,to_target",relative_ball_pos[10,:],to_target[10,:])
    # print("reward",to_target[10,:])
    # target_tray_pos = torch.tensor([ 0.4329, -0.2656,  1.0310], device=tray_positions.device)

    # print("center_pos:", center_pos[10,...])
    # print("relative_center_pose:", relative_center_pose[10,...])
    # print("rotated_offset:", rotated_offset[10,...])
    # print("relative_ball_pos:ball_position", relative_ball_pos[10,...],ball_positions[10,:])
    ball_velocities = quat_rotate(quat_conjugate(tray_orientations), ball_velocities)
    reset_dist_to_target= torch.sqrt(to_target[..., 0]  * to_target[..., 0] +
                           to_target[..., 1]  * to_target[..., 1] )
    dist_to_target= torch.sqrt(to_target[..., 0]  * to_target[..., 0] +
                           to_target[..., 1]  * to_target[..., 1]  + to_target[..., 2]  * to_target[..., 2] )
    # print(dist_to_target)
    ball_speed = torch.sqrt(ball_velocities[..., 0] * ball_velocities[..., 0] +
                            ball_velocities[..., 1] * ball_velocities[..., 1] +
                            ball_velocities[..., 2] * ball_velocities[..., 2])
   
    pos_reward = 1.0 / (1.0 + dist_to_target)
    speed_reward = 1.0 / (1.0 + ball_speed)
    
    tray_deviation =torch.sqrt(tray_positions[..., 0] * tray_positions[..., 0] +
                                 tray_positions[..., 2] * tray_positions[..., 2] +
                                 tray_positions[..., 1] * tray_positions[..., 1])
    tray_reward = 1.0 / (1.0 + tray_deviation)
    
    # print(smoothness_penalty)

    vertical_speed_penalty = torch.abs(ball_velocities[..., 0])   
    jump_threshold = 0.03
    jump_penalty = torch.where(
        to_target[..., 0] > jump_threshold,
        (to_target[..., 0] - jump_threshold) * 5.0,  
        torch.zeros_like(to_target[..., 0])
)   
    center_stability_bonus = torch.exp(-5.0 * dist_to_target)  

    # print("Before third_joint_penalty - action_buffer.shape:", action_buffer.shape)
    # third_joint_penalty = torch.mean(torch.abs(action_buffer[..., 2]), dim=(1))
    # print("After third_joint_penalty - action_buffer.shape:", action_buffer.shape)

    # print("Position Reward:", pos_reward[10])
    # print("Speed Reward:", speed_reward[10])
    # print("Tray Reward:", tray_reward[10])
    # print("Smoothness Penalty:", smoothness_penalty[10])
    # print("center_stability_bonus:", center_stability_bonus[10])

    reward = pos_reward * speed_reward  -smoothness_penalty# + 0.02*center_stability_bonus#-2*third_joint_penalty #-vertical_speed_penalty  - ball_heigh*jump_penalty
    # print(to_target[10])
    # print(dist_to_target[10] )
    
    # board_center_dist = torch.sqrt(ball_positions[..., 0] ** 2 + ball_positions[..., 1] ** 2 + ball_positions[..., 2] ** 2)
    # out_of_range = board_center_dist > 0.3
    #print(ball_positions[..., 1])
    # reset_condition = (torch.sqrt(ball_positions[..., 1] * ball_positions[..., 1]) <= 0.3) | (torch.sqrt(ball_positions[..., 0] * ball_positions[..., 0]) <= 0.3)
    # reset = torch.where(out_of_range, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    # reset = torch.where(ball_positions[..., 2] < ball_radius * 1.5, torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.sqrt(ball_positions[..., 1] * ball_positions[..., 1]) <= 0.3, torch.ones_like(reset_buf), reset_buf)
    # reset = torch.where(torch.sqrt(ball_positions[..., 1] * ball_positions[..., 1]) <= 0.3, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where( dist_to_target > 0.1  , torch.ones_like(reset_buf), reset)
    # reset = torch.where(ball_positions[..., 2] < ball_radius * 1.5, torch.ones_like(reset_buf), reset)
    # reset = torch.where(torch.abs(ball_positions[..., 0]) >  0.3  , torch.ones_like(reset_buf), reset)
    # reset = reset +1 < -1000

    return reward, reset
