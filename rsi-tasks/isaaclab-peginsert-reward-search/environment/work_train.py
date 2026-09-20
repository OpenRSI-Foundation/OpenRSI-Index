#!/usr/bin/env python3
import argparse, json, os, pathlib, re, sys
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(); parser.add_argument("--recipe",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--device",default="cuda:0",help="one Work GPU, cuda:0 or cuda:1; run up to two trials in parallel on distinct devices"); args=parser.parse_args()
if not re.fullmatch(r"cuda:[0-1]",args.device): raise SystemExit("device must be cuda:0 or cuda:1")
recipe_path=os.path.abspath(args.recipe); output=os.path.abspath(args.output_dir)
if not output.startswith("/workspace/research/"): raise SystemExit("output-dir must be below /workspace/research")
from reward_graph import load_recipe
recipe=load_recipe(recipe_path); pathlib.Path(output).mkdir(parents=True,exist_ok=True)
app=AppLauncher(headless=True,device=args.device,multi_gpu=False).app  # one Kit per GPU; Kit multi-GPU aborts on IOMMU hosts
try:
    import gymnasium as gym, math
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner
    from isaaclab_rl.rl_games import RlGamesGpuEnv,RlGamesVecEnvWrapper
    import isaaclab_tasks
    from isaaclab_tasks.direct.factory import factory_utils
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import isaacsim.core.utils.torch as torch_utils
    import torch
    torch.cuda.set_device(args.device)
    from driver_common import configure_assets,configure_task,fixed_agent_config,install_graph
    task="Isaac-Factory-PegInsert-Direct-v0"; env_cfg=load_cfg_from_registry(task,"env_cfg_entry_point"); agent_cfg=load_cfg_from_registry(task,"rl_games_cfg_entry_point")
    env_cfg.scene.num_envs=128; env_cfg.sim.device=args.device; env_cfg.seed=0; configure_assets(env_cfg); configure_task(env_cfg,json.loads(pathlib.Path("/opt/peginsert_public/fixed_env.json").read_text(encoding="utf-8")))
    gym_env=gym.make(task,cfg=env_cfg); from reward_graph import Runtime; install_graph(gym_env,recipe,Runtime,torch,factory_utils,torch_utils)
    wrapped=RlGamesVecEnvWrapper(gym_env,args.device,math.inf,1.0)
    vecenv.register("IsaacRlgWrapper",lambda config_name,num_actors,**kw:RlGamesGpuEnv(config_name,num_actors,**kw)); env_configurations.register("rlgpu",{"vecenv_type":"IsaacRlgWrapper","env_creator":lambda **kw:wrapped})
    agent_cfg=fixed_agent_config(agent_cfg,output,args.device); runner=Runner(IsaacAlgoObserver()); runner.load(agent_cfg); runner.reset(); runner.run({"train":True,"play":False,"sigma":None})
    (pathlib.Path(output)/"summary.json").write_text(json.dumps({"status":"complete","epochs":50,"transitions":819200,"recipe":recipe_path,"device":args.device})+"\n")
    gym_env.close()
finally: app.close()  # ends the process
