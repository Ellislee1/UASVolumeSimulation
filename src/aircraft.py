import numpy as np
from numpy import cos, sin, sqrt
import src.configs.config as Config

class Aircraft:
    def __init__(self, id, pos, spd, heading, start_time=0, route = None):
        self.id = id
        self.position = np.array(pos, dtype=np.float32)
        self.speed = spd
        self.heading = heading
        vx,vy = self.speed*cos(self.heading), self.speed*sin(heading)
        self.velocity = np.array([vx,vy], dtype=np.float32)

        self.start_time = start_time

        self.route = route

        self.next_wpt = self.route.route[1]
        self.terminal_wpt = self.route.route[-1]

        self.load_config()

    def load_config(self):
        self.G = Config.G
        self.scale = Config.scale
        self.min_speed = Config.min_speed
        self.max_speed = Config.max_speed
        self.speed_sigma = Config.speed_sigma
        self.d_heading = Config.d_heading
    
    def step(self, dh = 1):
        self.speed = max(self.min_speed, min(self.speed, self.max_speed))
        self.speed += np.random.normal(0, self.speed_sigma)
        self.heading += (dh - 1) * self.d_heading + np.random.normal(0, Config.heading_sigma)  # change heading
        vx = self.speed * cos(self.heading)
        vy = self.speed * sin(self.heading)
        self.velocity = np.array([vx, vy])

        self.position += self.velocity

    
    def genRelativeState(self, focus, dist):
        print(self.id, focus.id, focus.position - self.position, dist, [self.speed - focus.speed])
            
    
    def getTerminalSuccess(self):
        return self.terminal_wpt.getInBound(self.position)

    
    def copy(self):
        return Aircraft(self.id, self.position, self.speed, self.heading, self.start_time)
