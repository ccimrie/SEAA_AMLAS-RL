# All files needed to reproduce results in SEAA submitted paper Assuring Reinforcement Learning Components: AMLAS-RL.

The data generated and used for producing the results are included in the repo already. If you would like to also do the initial training of the agent you can run the appropriate files, but this will overwrite the exisiting saved data/models. 

## Adding custom environments
New environments had to be constructed to include minor changes. These are found in the safety_gym_envs; it is recommended to copy these into safety_gymnasium/tasks/safe_navigation/goal and make changes in the following files:

Adapt line 135 in <code>safety_gymnasium/__init__.py</code>:

<code>goal_tasks = {'Goal0': {}, 'Goal1': {}, 'Goal2': {}}</code> to <code>goal_tasks = {'Goal0': {}, 'Goal1': {}, 'Goal2': {}, 'GoalNormal1': {}, 'GoalCloseTest1': {}}</code>

Add the following imports <code>safety_gymnasium/tasks/__init__.py</code>:

<code>from safety_gymnasium.tasks.safe_navigation.goal.goal_normal import GoalNormalLevel1</code>

<code>from safety_gymnasium.tasks.safe_navigation.goal.goal_close_test import GoalCloseTestLevel1</code>
