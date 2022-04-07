import gym
from gym import spaces

class GymEnv(gym.Env):
    def __init__(self):
        super(GymEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects    
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(0, 15, shape=(2,5))

        print(self.observation_space.sample())



    def step(self, action):
        # Execute one time step within the environment
        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        pass

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

GymEnv()
