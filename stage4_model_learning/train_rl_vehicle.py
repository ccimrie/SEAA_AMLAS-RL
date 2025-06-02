import safety_gymnasium
from gymnasium.wrappers import TimeLimit
from DDPG.ddpg import DDPG
import numpy as np
import matplotlib.pyplot as plt 
import time
import os
import yaml

def trainAgent(yaml_file, reward_filename, seed):
    env_id = 'SafetyCarGoalNormal1-v0'
    reward_gain=r_ind
    
    cost_hazard_gain=c1_ind
    cost_vase_gain=c2_ind

    TT=500

    env=safety_gymnasium.make(env_id, max_episode_steps=TT) 
    env.task.mechanism_conf.continue_goal=False
    ddpg_controller=DDPG(yaml_file)
    start_ind=0
    reward_timecourse=[]

  ## Index for range sensors
    state_start_ind=24

    e=0
    E_max=500
  ## Allows start and stop of training
    reward_temp=len(np.loadtxt(reward_filename)) if os.path.exists(reward_filename) else 0
    E=E_max-reward_temp

  ## Main trianing loop
    while e<E:
        new_state=env.reset()
        obs=new_state[0][state_start_ind:]
        done=False
        truncated=False
        crashed=False
        t=0
        reward_max=0

        u_t=0
        while not done and not truncated and not crashed:
            cost_hazard=0
            cost_vase=0
            if t%50==0:
                print(f"    - {t}/{TT}")

            act=ddpg_controller.step(obs)
            new_state=env.step(act)
            obs_new=new_state[0][state_start_ind:]
            reward=new_state[1]
            cost=new_state[2]

            done=new_state[3]
            truncated=new_state[4]

            ## Check if crashed into obstacle
            if cost>0.9:
                cost_vase=1
                crashed=True
                if cost-1.0>0:
                    cost_hazard=0.2
                    u_t+=1
            elif cost>0:
                cost_hazard=0.2
                u_t+=1

            reward_total=reward*reward_gain-cost_hazard*cost_hazard_gain-cost_vase*cost_vase_gain
            ddpg_controller.recordStep(obs, act, reward_total, obs_new)
            
            ddpg_controller.learn()

            reward_max+=reward_total
            obs=obs_new
            t+=1

            ddpg_controller.saveNets()
        
      ## Record the reward
        if os.path.exists(reward_filename):
            permission='a'
        else:
            permission='w'
        with open(reward_filename, permission) as f:
            f.write(str(reward_max)+'\n')
            f.close()

      ## Record the outcome (reached goal + time, collided with obstacle, average time spent in unsafe zones)
        tracked_vals_dir="training_timecourse/tracked_training_values/"
        episode_outcome_filename=tracked_vals_dir+"e_outcome.txt"
        u_time_filename=tracked_vals_dir+"u_time.txt"
        if os.path.exists(episode_outcome_filename):
            permission='a'
        else:
            permission='w'
        with open(episode_outcome_filename, permission) as f:
            out=0
            if done:
                out=1
            elif crashed:
                out=-1
            outcome=f"{out}  {t}"
            f.write(outcome+'\n')
            f.close()
        if os.path.exists(u_time_filename):
            permission='a'
        else:
            permission='w'
        with open(u_time_filename, permission) as f:
            f.write(str(u_t)+'\n')
            f.close()
        e+=1
    return reward_max

### Make YAML file
yaml_filename='yaml_files/ddpg_architecture.yaml'
r_filename='training_timecourse/rewards/rewards.txt'
    
seed=np.random.randint(0,10e6)

rewards=trainAgent(yaml_filename, r_filename, seed)