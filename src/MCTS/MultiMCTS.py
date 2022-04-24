"""
MCTS approach addapted from the work by Xuxi Yang (https://github.com/xuxiyang1993/Multi_MCTS_Guidance_Separation_Assurance/blob/master/MCTS/common.py)
"""

import copy
import math

import numpy as np
import src.configs.config as C
from src.MCTS.MCTSbase import MCTSNode, MCTSState


class MultiAircraftState(MCTSState):
    def __init__(self, state, index, init_action, hit_wall = False, 
    conflict = False, reach_goal = False, prev_action = None, 
    depth = 0):
        MCTSState.__init__(self, state)
        self.index = index
        self.init_action = init_action
        self.hit_wall = hit_wall
        self.conflict = conflict
        self.reach_goal = reach_goal
        self.prev_action = prev_action
        self.depth = depth

        self.G = C.G
        self.scale = C.scale

        self.nearest_x = -1
        self.nearest_y = -1

    @property
    def reward(self):
        if self.hit_wall or self.conflict:
            return 0
        
        if self.reach_goal:
            return 1
        
        r = 1 - self.dist_goal() / 1200.0

        return r/4

    def move(self, a):
        next_state = self._move(a)
    
    def _move(self, a):
        state = copy.deepcopy(self.state)
        hit_wall = False
        conflict = False
        reach_goal = False

        for _ in range(C.simulate_frame):
            for i in range(state.shape[0]):
                heading = state[i,5] + (a[i] - 1) * C.d_heading
                speed = state[i, 4] + np.random.normal(0, C.speed_sigma)
                speed = max(C.min_speed, min(speed, C.max_speed))

                vx = speed * math.cos(heading)
                vy = speed * math.sin(heading)

                state[i, 0] += vx
                state[i, 1] += vy

                state[i, 2] = vx
                state[i, 3] = vy

                state[i, 4] = speed
                state[i, 5] = heading

            own_x = state[self.index][0]
            own_y = state[self.index][1]
            goal_x = state[self.index][6]
            goal_y = state[self.index][7]

            if not 0 < own_x < C.window_width or not 0 < own_y < C.window_height:
                hit_wall = True
                break

            if self.dist_intruder(state, own_x, own_y) < C.min_sep:
                conflict = True
                break

            if self.metric(own_x, own_y, goal_x, goal_y) < C.goal_radius:
                reach_goal = True

        return MultiAircraftState(state, self.index, 'random', hit_wall, conflict, reach_goal, a, self.depth+1)
    
    def get_legal_actions(self):
        return[0, 1, 2]
    
    def dist_goal(self):
        return self.metric(self.own_x, self.own_y, self.goal_x, self.goal_y)

    def dist_intruder(self, state, own_x, own_y):
        """
        I think I've already done this!!!!!
        """
        distance = 5000

        for i in [x for x in range(state.shape[0]) if x != self.index]:
            other_x = state[i][0]
            other_y = state[i][1]
            dist = self.metric(own_x, own_y, other_x, other_y)

            if dist < distance:
                distance = dist
                self.nearest_x = other_x
                self.nearest_y = other_y
        
        return distance

    def metric(self, x1, y1, x2, y2):
        dx = x1-x2
        dy = y1-y2
        return math.sqrt(dx**2 + dy**2)

    @property
    def ownx(self):
        return self.state[self.index][0]

    @property
    def owny(self):
        return self.state[self.index][1]

    @property
    def goalx(self):
        return self.state[self.index][6]

    @property
    def goaly(self):
        return self.state[self.index][7]

    def __repr__(self) -> str:
        s = 'index: %d, prev action: %s, pos: %.2f,%.2f, goal: %.2f,%.2f, dist goal: %.2f, dist intruder: %f,' \
            'nearest intruder: (%.2f, %.2f), depth: %d' \
            % (self.index,
               self.prev_action,
               self.ownx,
               self.owny,
               self.goalx,
               self.goaly,
               self.dist_goal(),
               self.dist_intruder(self.state, self.ownx, self.owny),
               self.nearest_x,
               self.nearest_y,
               self.depth)
        return s

class MultiAircraftNode(MCTSNode):
    def __init__(self, state: MultiAircraftState, parent = None):
        MCTSNode.__init__(self,parent)
        self.state = state

    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.get_legal_actions()
        
        return self._untried_actions

    @property
    def reward(self):
        return self.q / self.n if self.n else 0

    def expand(self):
        a = self.untried_actions.pop()

        if self.state.init_action == 'random':
            all_action = np.random.randint(0,3, size = self.state.state.shape[0])
        else:
            all_action = self.state.init_action.copy()
        
        all_action[self.state.index] = a

        next_state = self.state.move(all_action)
        child_node = MultiAircraftNode(next_state, parent = self)

        self.children.append(child_node)
        return child_node
    
    def is_terminal_node(self, search_depth):
        return self.state.is_terminal_state(search_depth)
    
    def rollout(self, search_depth):
        current_rollout_state = self.state

        while not current_rollout_state.is_terminal_state(search_depth):
            action = np.random.randint(0, 3, size = self.state.state.shape[0])
            current_rollout_state = current_rollout_state.move(action)
        
        return current_rollout_state.reward()

    def backpropagate(self, result):
        self.n += 1
        self.q += result

        if self.parent:
            self.parent.backpropagate(result)

    def __repr__(self) -> str:
        s = 'Agent: %d, Node: children: %d; visits: %d; reward: %.4f; p_action: %s, state: (%.2f, %.2f); ' \
            'goal: (%.2f, %.2f), dist: %.2f, nearest: (%.2f, %.2f)' \
            % (self.state.index + 1,
               len(self.children),
               self.n,
               self.q / (self.n + 1e-2),
               self.state.prev_action,
               self.state.ownx,
               self.state.owny,
               self.state.goalx,
               self.state.goaly,
               self.state.dist_goal(),
               self.state.nearest_x,
               self.state.nearest_y)

        return s