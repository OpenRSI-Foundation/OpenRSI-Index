"""Source-derived hooks shared only by public Work tooling."""
import types

def configure_assets(env_cfg):
    root="/opt/peginsert_assets/Isaac"
    import isaaclab_tasks.direct.factory.factory_env as factory_env
    ground_cfg=factory_env.GroundPlaneCfg
    ground=f"{root}/Environments/Grid/default_environment.usd"
    factory_env.GroundPlaneCfg=lambda: ground_cfg(usd_path=ground)
    factory_env.ISAAC_NUCLEUS_DIR=root
    env_cfg.robot.spawn.usd_path=f"{root}/IsaacLab/Factory/franka_mimic.usd"
    env_cfg.task.robot_cfg.robot_usd=env_cfg.robot.spawn.usd_path
    env_cfg.task.fixed_asset_cfg.usd_path=f"{root}/IsaacLab/Factory/factory_hole_8mm.usd"
    env_cfg.task.held_asset_cfg.usd_path=f"{root}/IsaacLab/Factory/factory_peg_8mm.usd"
    env_cfg.task.fixed_asset.spawn.usd_path=env_cfg.task.fixed_asset_cfg.usd_path
    env_cfg.task.held_asset.spawn.usd_path=env_cfg.task.held_asset_cfg.usd_path

def configure_task(env_cfg,fixed_env):
    """Apply the fixed environment constants (episode length and reset randomization) from fixed_env.json."""
    if fixed_env.get("schema_version")!=1: raise RuntimeError("unsupported fixed_env schema")
    env_cfg.episode_length_s=float(fixed_env["episode_length_s"])
    for key,value in fixed_env["task"].items():
        if not hasattr(env_cfg.task,key): raise RuntimeError(f"unknown fixed task field: {key}")
        setattr(env_cfg.task,key,[float(x) for x in value] if isinstance(value,list) else float(value))

def install_graph(env, recipe, runtime_cls, torch, factory_utils, torch_utils):
    base=env.unwrapped; runtime=runtime_cls(recipe,torch,base.num_envs,base.device); base.task_reward_runtime=runtime; base.task_terminal_success=None
    original_reset=base._reset_buffers
    original_rewards=base._get_rewards
    def reset_buffers(self,ids):
        original_reset(ids)
        if self.task_reward_runtime is not None: self.task_reward_runtime.reset(ids)
    def rewards(self):
        if self.task_reward_runtime is None:
            result=original_rewards()
            if bool((self.episode_length_buf>=self.max_episode_length-1).all()): self.task_terminal_success=self._get_curr_successes(self.cfg_task.success_threshold,False).detach().clone()
            return result
        success=self._get_curr_successes(self.cfg_task.success_threshold,False)
        engaged=self._get_curr_successes(self.cfg_task.engage_threshold,False)
        held_pos,held_quat=factory_utils.get_held_base_pose(self.held_pos,self.held_quat,self.cfg_task.name,self.cfg_task.fixed_asset_cfg,self.num_envs,self.device)
        target_pos,target_quat=factory_utils.get_target_held_base_pose(self.fixed_pos,self.fixed_quat,self.cfg_task.name,self.cfg_task.fixed_asset_cfg,self.num_envs,self.device)
        offsets=factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints,self.device)*self.cfg_task.keypoint_scale
        held=torch.stack([torch_utils.tf_combine(held_quat,held_pos,torch.tensor([1.,0.,0.,0.],device=self.device).repeat(self.num_envs,1),offset.repeat(self.num_envs,1))[1] for offset in offsets],1)
        fixed=torch.stack([torch_utils.tf_combine(target_quat,target_pos,torch.tensor([1.,0.,0.,0.],device=self.device).repeat(self.num_envs,1),offset.repeat(self.num_envs,1))[1] for offset in offsets],1)
        keypoint=torch.linalg.vector_norm(held-fixed,dim=-1).mean(-1); scalar=lambda value: torch.full_like(keypoint,float(value))
        inputs={"peg_position":held_pos,"peg_quaternion":held_quat,"peg_linear_velocity":self._held_asset.data.root_lin_vel_w.clone(),"peg_angular_velocity":self._held_asset.data.root_ang_vel_w.clone(),"hole_position":target_pos,"hole_quaternion":target_quat,"hole_linear_velocity":self._fixed_asset.data.root_lin_vel_w.clone(),"hole_angular_velocity":self._fixed_asset.data.root_ang_vel_w.clone(),"fingertip_position":self.fingertip_midpoint_pos,"fingertip_quaternion":self.fingertip_midpoint_quat,"fingertip_linear_velocity":self.fingertip_midpoint_linvel,"fingertip_angular_velocity":self.fingertip_midpoint_angvel,"joint_position":self.joint_pos[:,:7],"joint_velocity":self.joint_vel[:,:7],"action":self.actions,"previous_action":self.prev_actions,"peg_diameter":scalar(self.cfg_task.held_asset_cfg.diameter),"peg_height":scalar(self.cfg_task.held_asset_cfg.height),"hole_diameter":scalar(self.cfg_task.fixed_asset_cfg.diameter),"hole_height":scalar(self.cfg_task.fixed_asset_cfg.height),"episode_progress":self.episode_length_buf.float()/float(self.max_episode_length),"keypoint_distance":keypoint,"engaged":engaged.float(),"success":success.float()}
        reward=self.task_reward_runtime.evaluate(inputs); self.prev_actions=self.actions.clone(); self._log_factory_metrics({"candidate":reward},success)
        if bool((self.episode_length_buf>=self.max_episode_length-1).all()): self.task_terminal_success=success.detach().clone()
        return reward
    base._reset_buffers=types.MethodType(reset_buffers,base); base._get_rewards=types.MethodType(rewards,base)

def fixed_agent_config(agent_cfg,output,device="cuda:0"):
    params=agent_cfg["params"]; params["seed"]=0; params["load_checkpoint"]=False; params["load_path"]=""
    cfg=params["config"]; cfg.update({"device":device,"device_name":device,"multi_gpu":False,"num_actors":128,"horizon_length":128,"max_epochs":50,"save_frequency":0,"save_best_after":51,"train_dir":output,"full_experiment_name":"fixed"}); cfg.pop("score_to_win",None)
    return agent_cfg
