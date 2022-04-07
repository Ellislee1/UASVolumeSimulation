from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
from numpy import arctan2, pi
from numpy import sin, cos
import numpy as np

from src.functions import getDistance

class Volume:
    def __init__(self, id, points):
        self.id = id
        self.points = points
        self.area = Polygon(points)
    
    def getInVolume(self, active_dict):
        in_volume = []

        for _, ac in active_dict.items():
            position = Point(ac.position)
            if self.area.contains(position):
                in_volume.append(ac.id)
        
        return in_volume
    
    def getPointIn(self, point):
        position = Point(point)
        return self.area.contains(position)
    
    def getNearestIntersection(self, position, heading):
        length = 10000
        min_dist, closest = np.inf, None

        endy = position[1] + length*sin(heading)
        endx = position[0] + length*cos(heading)

        ac_line = LineString([position, [endx,endy]])

        for i in range(len(self.points)):
            segment = LineString([self.points[i-1], self.points[i]])
            intersect = segment.intersection(ac_line)
            if not isinstance(intersect, LineString):
                x,y = intersect.coords.xy
                x,y = x[0], y[0]
                dist = getDistance(position, [x,y])
                if  dist < min_dist:
                    min_dist = dist
                    closest = intersect


class Waypoint:
    def __init__(self, id, position, spawn = True):
        self.id = id
        self.position = position
        self.volumes = set()
        self.spawn = spawn

        self.area_box = Polygon([[self.position[0]-25,self.position[1]-25], [self.position[0]-25,self.position[1]+25],[self.position[0]+25,self.position[1]+25],[self.position[0]+25,self.position[1]-25]])

    def add_volume(self, volume):
        self.volumes.add(volume)
    
    def getInBound(self, position):
        return self.area_box.contains(Point(position))

class Route:
    def __init__(self, id, route_wpts):
        self.id = id
        self.route = route_wpts
        self.getInitialHeading(self.route[0].position,self.route[1].position)

    def getInitialHeading(self, start, _next):
        theta = arctan2(_next[0]-start[0],_next[1]-start[1])-(pi/2)

        self.theta = theta if theta >= 0 else ((2*pi) + theta)
