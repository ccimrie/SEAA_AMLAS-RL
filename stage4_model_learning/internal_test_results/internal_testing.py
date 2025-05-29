import safety_gymnasium
from gymnasium.wrappers import TimeLimit
from DDPG.ddpg import DDPG
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
import yaml
from multiprocessing import Pool
#import tensorflow as tf

def testAgent(yaml_file, test_results_file, seed):
    env_id = 'SafetyCarGoal1-v0'
    energy=250
    # render_mode='human'
    render_mode=''
    env=safety_gymnasium.make(env_id, max_episode_steps=energy, render_mode=render_mode)
    env.task.mechanism_conf.continue_goal=False
    env.set_seed(seed)

    ddpg_controller=DDPG(yaml_file)

    TT=1000
    if os.path.exists(test_results_file):
        test_results=np.load(test_results_file)['results']
        tt=len(test_results)
    else:
        ## Recording: 
        #   - Success 
        #   - energy left 
        #   - Crashed 
        #   - time spent in danger zone
        test_results=np.empty((0,4))
        tt=0
    state_start_ind=24
    while tt<TT:
        print(f"  - Simulation {tt}/{TT}")
        new_state=env.reset()
        obs=new_state[0][state_start_ind:]
        done=False
        truncated=False
        crashed=False
        danger_time=0
        t=0
        while not done and not truncated and not crashed:
            act=ddpg_controller.act(obs)
            new_state=env.step(act)
            obs_new=new_state[0][state_start_ind:]

            reward=new_state[1]
            cost=new_state[2]
            done=new_state[3]
            truncated=new_state[4]

            ## Check if crashed into obstacle
            if cost>0.9:
                crashed=True
                if cost-1>0: ## Check if in danger zone
                    danger_time+=1
            elif cost>0: ## Check if in danger zone
                danger_time+=1
            obs=obs_new
            t+=1
        test_result=np.array([float(done), energy-t, float(crashed), danger_time])
        test_results=np.vstack((test_results, test_result))
        np.savez(test_results_file[:-4], results=test_results)
        tt+=1
    return test_results

yaml_filename='../yaml_files/ddpg_architecture.yaml'
test_results_filename='results.npz'
seed=np.random.randint(0,10e6)

testAgent(yaml_filename,test_results_filename,seed)