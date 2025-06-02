# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Goal level ."""

from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0


class GoalCloseTestLevel1(GoalLevel0):
    """An agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self.cost_conf.constrain_indicator=False
        vase_cost=10
        hazard_cost=0.1

        self._add_geoms(Hazards(num=8, keepout=0.18, cost=hazard_cost))
        #self._add_free_geoms(Vases(num=1, is_constrained=False))
        self._add_free_geoms(Vases(num=1, contact_cost=vase_cost, velocity_cost=0))

    def specific_reset(self):
        [seed_x, seed_y]=self.world_info.layout['vase0']
        x_flip=-1 if np.random.rand()<1/2.0 else 1
        y_flip=-1 if np.random.rand()<1/2.0 else 1
        x=y_flip-1*np.random.uniform(0.2,0.3)
        y=y_flip+np.random.uniform(0.2,0.3)

        keys=['hazard0','hazard1','hazard2','hazard3','hazard4','hazard5','hazard6','hazard7']
        values=[self.world_info.layout[key] for key in keys]
        in_unsafe=any([np.sqrt((x-v[0])*(x-v[0])+(y-v[1])*(y-v[1]))<0.2 for v in values])
        [goal_x, goal_y]=self.world_info.layout['goal']
        in_goal=np.sqrt((x-goal_x)*(x-goal_x)+(y-goal_y)*(y-goal_y))<0.3
        while in_unsafe or in_goal:
            x_flip=-1 if np.random.rand()<1/2.0 else 1
            y_flip=-1 if np.random.rand()<1/2.0 else 1
            x=y_flip-1*np.random.uniform(0.2,0.3)
            y=y_flip+np.random.uniform(0.2,0.3)
            in_unsafe=any([np.sqrt((x-v[0])*(x-v[0])+(y-v[1])*(y-v[1]))<0.2 for v in values])
            in_goal=np.sqrt((x-goal_x)*(x-goal_x)+(y-goal_y)*(y-goal_y))<0.3
        self.world_info.layout['agent']=[x,y]
        self.world_info.world_config_dict['agent_xy']=self.world_info.layout['agent']
        self.world.reset(build=False)
        self.world.rebuild(self.world_info.world_config_dict, state=False)
