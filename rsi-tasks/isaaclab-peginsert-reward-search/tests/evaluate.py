#!/usr/bin/env python3
"""Ordinary scoring route: validate, fresh-train, retain, and evaluate up to two recipes, one per Judge GPU.

The parent process stages and validates every candidate, then runs one worker
process per candidate pinned to its own GPU (`cuda:0` or `cuda:1`). Each worker
is the unchanged single-GPU fixed protocol. The submission reward is the best
candidate score; every candidate score is reported alongside it.
"""
from __future__ import annotations
import importlib.util, json, math, os, pathlib, subprocess, sys, tempfile, time, traceback, types

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load fixed module: {path}")
    item=importlib.util.module_from_spec(spec); sys.modules[name]=item; spec.loader.exec_module(item); return item

bootstrap=module("judge_bootstrap","/tests/bootstrap.py")

def install_graph(env,recipe,runtime_cls,torch,factory_utils,torch_utils):
    base=env.unwrapped; runtime=runtime_cls(recipe,torch,base.num_envs,base.device); base.task_reward_runtime=runtime; base.task_terminal_success=None
    original_reset=base._reset_buffers
    original_rewards=base._get_rewards
    original_reset_idx=base._reset_idx
    original_ik=base.set_pos_inverse_kinematics
    base.task_reset_stats={"batches":0,"environments":0,"ik_calls":0,"ik_rejections":0}
    def reset_buffers(self,ids):
        original_reset(ids)
        if self.task_reward_runtime is not None: self.task_reward_runtime.reset(ids)
    def reset_idx(self,ids): self.task_reset_stats["batches"]+=1; self.task_reset_stats["environments"]+=int(ids.numel()); return original_reset_idx(ids)
    def inverse_kinematics(self,*args,**kwargs):
        result=original_ik(*args,**kwargs); rejected=torch.logical_or(torch.linalg.vector_norm(result[0],dim=1)>1e-3,torch.linalg.vector_norm(result[1],dim=1)>1e-3)
        self.task_reset_stats["ik_calls"]+=1; self.task_reset_stats["ik_rejections"]+=int(rejected.sum().item()); return result
    def rewards(self):
        if self.task_reward_runtime is None:
            result=original_rewards()
            if bool((self.episode_length_buf>=self.max_episode_length-1).all()): self.task_terminal_success=self._get_curr_successes(self.cfg_task.success_threshold,False).detach().clone()
            return result
        success=self._get_curr_successes(self.cfg_task.success_threshold,False); engaged=self._get_curr_successes(self.cfg_task.engage_threshold,False)
        held_pos,held_quat=factory_utils.get_held_base_pose(self.held_pos,self.held_quat,self.cfg_task.name,self.cfg_task.fixed_asset_cfg,self.num_envs,self.device)
        target_pos,target_quat=factory_utils.get_target_held_base_pose(self.fixed_pos,self.fixed_quat,self.cfg_task.name,self.cfg_task.fixed_asset_cfg,self.num_envs,self.device)
        offsets=factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints,self.device)*self.cfg_task.keypoint_scale; identity=torch.tensor([1.,0.,0.,0.],device=self.device).repeat(self.num_envs,1)
        held=torch.stack([torch_utils.tf_combine(held_quat,held_pos,identity,offset.repeat(self.num_envs,1))[1] for offset in offsets],1); fixed=torch.stack([torch_utils.tf_combine(target_quat,target_pos,identity,offset.repeat(self.num_envs,1))[1] for offset in offsets],1)
        keypoint=torch.linalg.vector_norm(held-fixed,dim=-1).mean(-1); scalar=lambda x: torch.full_like(keypoint,float(x))
        inputs={"peg_position":held_pos,"peg_quaternion":held_quat,"peg_linear_velocity":self._held_asset.data.root_lin_vel_w.clone(),"peg_angular_velocity":self._held_asset.data.root_ang_vel_w.clone(),"hole_position":target_pos,"hole_quaternion":target_quat,"hole_linear_velocity":self._fixed_asset.data.root_lin_vel_w.clone(),"hole_angular_velocity":self._fixed_asset.data.root_ang_vel_w.clone(),"fingertip_position":self.fingertip_midpoint_pos,"fingertip_quaternion":self.fingertip_midpoint_quat,"fingertip_linear_velocity":self.fingertip_midpoint_linvel,"fingertip_angular_velocity":self.fingertip_midpoint_angvel,"joint_position":self.joint_pos[:,:7],"joint_velocity":self.joint_vel[:,:7],"action":self.actions,"previous_action":self.prev_actions,"peg_diameter":scalar(self.cfg_task.held_asset_cfg.diameter),"peg_height":scalar(self.cfg_task.held_asset_cfg.height),"hole_diameter":scalar(self.cfg_task.fixed_asset_cfg.diameter),"hole_height":scalar(self.cfg_task.fixed_asset_cfg.height),"episode_progress":self.episode_length_buf.float()/float(self.max_episode_length),"keypoint_distance":keypoint,"engaged":engaged.float(),"success":success.float()}
        result=self.task_reward_runtime.evaluate(inputs); self.prev_actions=self.actions.clone(); self._log_factory_metrics({"candidate":result},success)
        if bool((self.episode_length_buf>=self.max_episode_length-1).all()): self.task_terminal_success=success.detach().clone()
        return result
    base._reset_buffers=types.MethodType(reset_buffers,base); base._reset_idx=types.MethodType(reset_idx,base); base.set_pos_inverse_kinematics=types.MethodType(inverse_kinematics,base); base._get_rewards=types.MethodType(rewards,base)

def configure_assets(env_cfg):
    import isaaclab_tasks.direct.factory.factory_env as factory_env
    root="/opt/peginsert_assets/Isaac"; ground_cfg=factory_env.GroundPlaneCfg; ground=f"{root}/Environments/Grid/default_environment.usd"; factory_env.GroundPlaneCfg=lambda: ground_cfg(usd_path=ground); factory_env.ISAAC_NUCLEUS_DIR=root
    env_cfg.robot.spawn.usd_path=f"{root}/IsaacLab/Factory/franka_mimic.usd"; env_cfg.task.robot_cfg.robot_usd=env_cfg.robot.spawn.usd_path
    env_cfg.task.fixed_asset_cfg.usd_path=f"{root}/IsaacLab/Factory/factory_hole_8mm.usd"; env_cfg.task.held_asset_cfg.usd_path=f"{root}/IsaacLab/Factory/factory_peg_8mm.usd"; env_cfg.task.fixed_asset.spawn.usd_path=env_cfg.task.fixed_asset_cfg.usd_path; env_cfg.task.held_asset.spawn.usd_path=env_cfg.task.held_asset_cfg.usd_path

def configure_task(env_cfg,fixed_env):
    """Apply the fixed environment constants (episode length and reset randomization) from fixed_env.json."""
    if fixed_env.get("schema_version")!=1: raise RuntimeError("unsupported fixed_env schema")
    env_cfg.episode_length_s=float(fixed_env["episode_length_s"])
    for key,value in fixed_env["task"].items():
        if not hasattr(env_cfg.task,key): raise RuntimeError(f"unknown fixed task field: {key}")
        setattr(env_cfg.task,key,[float(x) for x in value] if isinstance(value,list) else float(value))

def fixed_config(agent_cfg,cfg,scratch,device):
    params=agent_cfg["params"]; params["seed"]=cfg["train_seed"]; params["load_checkpoint"]=False; params["load_path"]=""
    run=params["config"]; run.update({"device":device,"device_name":device,"multi_gpu":False,"num_actors":cfg["num_envs"],"horizon_length":cfg["horizon"],"max_epochs":cfg["epochs"],"save_frequency":0,"save_best_after":cfg["epochs"]+1,"train_dir":str(scratch),"full_experiment_name":"judge"}); run.pop("score_to_win",None)
    return agent_cfg

def worker(mode,cfg,staged,device_index,result_path):
    """One candidate (or the released reference) on one GPU; writes the summary before Kit shuts the process down."""
    started=time.monotonic(); device=f"cuda:{device_index}"
    graph=module("judge_reward_graph","/tests/reward_graph.py")
    recipe=graph.load_recipe(str(staged)) if mode=="candidate" else None
    from isaaclab.app import AppLauncher
    # multi_gpu=False: one Kit per GPU (as Isaac Lab distributed training does); Kit's multi-GPU
    # foundation otherwise aborts on IOMMU hosts during its peer-to-peer validation.
    simulation=AppLauncher(headless=True,device=device,multi_gpu=False).app
    gym_env=None
    try:
        import gymnasium as gym, torch
        torch.cuda.set_device(device)
        from rl_games.common import env_configurations,vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner
        from isaaclab_rl.rl_games import RlGamesGpuEnv,RlGamesVecEnvWrapper
        import isaaclab_tasks
        from isaaclab_tasks.direct.factory import factory_utils
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        import isaacsim.core.utils.torch as torch_utils
        task=cfg["task"]; env_cfg=load_cfg_from_registry(task,"env_cfg_entry_point"); agent_cfg=load_cfg_from_registry(task,"rl_games_cfg_entry_point")
        env_cfg.scene.num_envs=cfg["num_envs"]; env_cfg.sim.device=device; env_cfg.seed=cfg["train_seed"]; configure_assets(env_cfg); configure_task(env_cfg,json.loads(pathlib.Path("/opt/peginsert_public/fixed_env.json").read_text(encoding="utf-8"))); gym_env=gym.make(task,cfg=env_cfg)
        runtime_recipe=recipe if recipe is not None else graph.load_recipe("/opt/peginsert_public/baseline_reward.json")
        install_graph(gym_env,runtime_recipe,graph.RewardRuntime,torch,factory_utils,torch_utils)
        if recipe is None: gym_env.unwrapped.task_reward_runtime=None
        wrapped=RlGamesVecEnvWrapper(gym_env,device,math.inf,1.0); count={"charged":0,"completed":0}; raw_step=wrapped.step
        def counted(actions):
            result=raw_step(actions); count["completed"]+=1; return result
        wrapped.step=counted
        vecenv.register("IsaacRlgWrapper",lambda config_name,num_actors,**kw:RlGamesGpuEnv(config_name,num_actors,**kw)); env_configurations.register("rlgpu",{"vecenv_type":"IsaacRlgWrapper","env_creator":lambda **kw:wrapped})
        agent_cfg=fixed_config(agent_cfg,cfg,staged.parent,device)
        if mode=="candidate":
            selected=recipe["selected_epoch"]
            class Snapshot(IsaacAlgoObserver):
                def __init__(self): super().__init__(); self.state=None; self.last=0
                def after_init(self,algo):
                    super().after_init(algo); original=algo.get_action_values
                    def charged(*args,**kwargs): count["charged"]+=1; return original(*args,**kwargs)
                    algo.get_action_values=charged
                def after_print_stats(self,frame,epoch,total):
                    if epoch<self.last: raise RuntimeError("training epoch regressed")
                    self.last=epoch
                    if epoch==selected and self.state is None: self.state={k:v.detach().cpu().clone() for k,v in self.algo.model.state_dict().items()}
                    super().after_print_stats(frame,epoch,total)
            observer=Snapshot(); runner=Runner(observer); runner.load(agent_cfg); runner.reset(); runner.run({"train":True,"play":False,"sigma":None})
            if observer.last!=cfg["epochs"] or observer.state is None or count["charged"]!=cfg["vector_steps"] or count["completed"]!=cfg["vector_steps"]: raise RuntimeError(f"fixed training accounting or selected snapshot incomplete: epochs {observer.last}/{cfg['epochs']}, snapshot at epoch {selected} {'taken' if observer.state is not None else 'missing'}, charged {count['charged']}/{cfg['vector_steps']}, completed {count['completed']}/{cfg['vector_steps']}")
            state=observer.state
        else:
            from safetensors.torch import load_file
            state=load_file("/opt/peginsert_reference/converted/policy.safetensors",device="cpu"); runner=Runner(); runner.load(agent_cfg)
        if len(state)!=28 or any(not bool(torch.isfinite(value).all().item()) for value in state.values()): raise RuntimeError("selected player tensor inventory mismatch")
        if mode=="reference":
            variances=[value for name,value in state.items() if "running_mean_std" in name and name.endswith("running_var")]
            if not variances or any(not bool((value>0).all().item()) for value in variances): raise RuntimeError("reference normalization state is invalid")
        player=runner.create_player(); player.model.load_state_dict(state,strict=True); player.model.eval(); player.is_deterministic=True
        successes=0; eval_steps=0; base=gym_env.unwrapped; base.task_reward_runtime=None; base.task_reset_stats={"batches":0,"environments":0,"ik_calls":0,"ik_rejections":0}
        policy_obs=lambda item: item["obs"] if isinstance(item,dict) else item  # the wrapper also returns critic states
        for seed in cfg["evaluation_seeds"]:
            base.actions.zero_(); base.prev_actions.zero_()
            wrapped.seed(seed); obs=policy_obs(wrapped.reset())
            if bool(torch.count_nonzero(base.actions).item()) or bool(torch.count_nonzero(base.prev_actions).item()): raise RuntimeError("evaluation controller reset is not canonical")
            player.get_batch_size(obs,1); player.init_rnn(); terminal=None
            for _ in range(base.max_episode_length+2):
                with torch.no_grad(): actions=player.get_action(player.obs_to_torch(obs),is_deterministic=True); obs,_,dones,_=raw_step(actions); obs=policy_obs(obs)
                eval_steps+=1
                if bool(torch.as_tensor(dones).all().item()): terminal=base.task_terminal_success.detach().clone(); break
            if terminal is None or terminal.numel()!=cfg["num_envs"]: raise RuntimeError("fixed evaluation episode batch incomplete")
            successes+=int(terminal.sum().item())
        if successes<0 or successes>cfg["evaluation_episodes"]: raise RuntimeError("invalid success total")
        reward=successes/cfg["evaluation_episodes"]
        if not math.isfinite(reward): raise RuntimeError("reward is not finite")
        summary={"status":"complete","reward":reward,"successes":successes,"failures":cfg["evaluation_episodes"]-successes,"episodes":cfg["evaluation_episodes"],"selected_epoch":recipe["selected_epoch"] if recipe else "released_reference","device":device,"completed_vector_steps":count["completed"],"completed_transitions":count["completed"]*cfg["num_envs"],"charged_vector_steps":count["charged"],"charged_transitions":count["charged"]*cfg["num_envs"],"evaluation_policy_steps":eval_steps,"evaluation_batches":len(cfg["evaluation_seeds"]),"reset_batches":base.task_reset_stats["batches"],"reset_environments":base.task_reset_stats["environments"],"ik_calls":base.task_reset_stats["ik_calls"],"ik_rejections":base.task_reset_stats["ik_rejections"],"elapsed_sec":round(time.monotonic()-started,3)}
        with open(result_path,"x",encoding="utf-8") as stream: json.dump(summary,stream,sort_keys=True); stream.flush(); os.fsync(stream.fileno())
        return summary
    except BaseException as error:
        # Record the failure here: the finally clause below ends the process, so no outer handler runs.
        with open(result_path+".error","w",encoding="utf-8") as stream: stream.write(traceback.format_exc())
        print(f"worker cuda:{device_index} failed: {type(error).__name__}: {error}",flush=True); traceback.print_exc(); raise
    finally:
        if gym_env is not None: gym_env.close()
        simulation.close()  # SimulationApp.close() ends the process; nothing after it runs

def worker_main(argv):
    mode,staged,device_index,result_path=argv[0],pathlib.Path(argv[1]),int(argv[2]),argv[3]
    cfg=json.loads(pathlib.Path("/tests/fixed_config.json").read_text())
    worker(mode,cfg,staged,device_index,result_path)
    return 0

def run(mode="candidate"):
    started=time.monotonic()
    try: cfg,staged=bootstrap.preflight()
    except bootstrap.CandidateProblem as exc: print(json.dumps(exc.envelope(),sort_keys=True),flush=True); return 2
    if mode=="reference": staged=staged[:1]
    if len(staged)>cfg["max_candidates"]: raise RuntimeError("candidate count exceeds the fixed Judge GPU count")
    graph=module("judge_reward_graph","/tests/reward_graph.py")
    if mode=="candidate":
        for public,path in staged:
            try: graph.load_recipe(str(path))
            except graph.CandidateInvalid as exc: print(json.dumps({**exc.envelope(),"path":public},sort_keys=True),flush=True); return 2
    import torch
    if torch.cuda.device_count()<len(staged): raise RuntimeError(f"Judge needs {len(staged)} GPUs, found {torch.cuda.device_count()}")
    results=pathlib.Path(tempfile.mkdtemp(prefix="peginsert-judge-results-",dir="/tmp")); os.chmod(results,0o700)
    procs=[]
    for index,(public,path) in enumerate(staged):
        result=results/f"candidate_{index}.json"
        print(json.dumps({"event":"worker_started","candidate":index,"path":public,"device":f"cuda:{index}"},sort_keys=True),flush=True)
        procs.append(subprocess.Popen([sys.executable,"-I","-u",__file__,"--worker",mode,str(path),str(index),str(result)],stdin=subprocess.DEVNULL))
    codes=[proc.wait() for proc in procs]
    summaries=[]
    for index,(public,path) in enumerate(staged):
        result=results/f"candidate_{index}.json"
        if codes[index]!=0 or not result.is_file():
            detail=pathlib.Path(str(result)+".error"); detail=detail.read_text(encoding="utf-8")[-4000:] if detail.is_file() else "no Python traceback (the simulator ended the worker process)"
            raise RuntimeError(f"candidate {index} worker failed (exit {codes[index]}): {detail}")
        summary=json.loads(result.read_text(encoding="utf-8"))
        if summary.get("status")!="complete" or not math.isfinite(float(summary["reward"])) or summary["completed_vector_steps"]!=(cfg["vector_steps"] if mode=="candidate" else 0): raise RuntimeError(f"candidate {index} summary is incomplete")
        summary["candidate"]=index; summary["path"]=public; summaries.append(summary)
        print(json.dumps(summary,sort_keys=True),flush=True)
    rewards={f"candidate_{s['candidate']}":float(s["reward"]) for s in summaries}
    best=max(summaries,key=lambda s: (float(s["reward"]),-s["candidate"]))
    print(json.dumps({"status":"complete","reward":float(best["reward"]),"best_candidate":best["candidate"],"best_path":best["path"],"candidates":len(summaries),"candidate_rewards":rewards,"elapsed_sec":round(time.monotonic()-started,3)},sort_keys=True),flush=True)
    if mode=="candidate":
        with open("/logs/verifier/reward.json","x",encoding="utf-8") as stream: json.dump({"reward":float(best["reward"]),**rewards},stream)
    return 0

def main():
    try:
        if len(sys.argv)>1 and sys.argv[1]=="--worker": return worker_main(sys.argv[2:])
        return run("candidate")
    except Exception:
        traceback.print_exc(); return 1
if __name__=="__main__": raise SystemExit(main())
