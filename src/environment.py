import contextlib
import os
from collections import OrderedDict

import numpy as np
from numpy import pi

from src.aircraft import Aircraft
from environments.cases import setupCase1, setupCase2


class Environment:
    def __init__(self, viewport = (800,800), separations = None, max_aircraft = 15, min_dist = 15, visual = False):
        if separations is None:
            separations = [30,40,50]


        self.viewport = viewport
        self.viewer = None
        self.visual = visual

        self.ac_dict = AircraftDict()
        self.volume_dict = VolumeDict()
        self.waypoint_dict = WaypointDict()

        self.volume_dict, self.waypoint_dict, self.routes = setupCase1(self.volume_dict, self.waypoint_dict)
        self.min_dist = min_dist


        self.num_aircraft = 0
        self.active = 0
        self.max_aircraft = max_aircraft
        self.counter = 0

        self.success, self.fail = 0, 0

        for id, volume in self.volume_dict.items():
            in_volume = volume.getInVolume(self.waypoint_dict.waypoint_dict)

            for wpt in in_volume:
                self.waypoint_dict.waypoint_dict[wpt].add_volume(id)

        self.separations = separations
        self.positions = []
        self.offsets = []
        for id, wpt in self.waypoint_dict.items():
            if wpt.spawn:
                self.positions.append(id)
                self.offsets.append(np.random.choice(self.separations,1)[0])
        self.offsets = np.array(self.offsets)


    def step(self):
        self.genAircraft()
        dist_matrix = self.ac_dict.genDistMatrix()
        ac_keys = np.array(list(self.ac_dict.ac_dict.keys()))

        for _,ac in self.ac_dict.ac_dict.items():
            i = np.argwhere(ac_keys == ac.id)[0]
            dist_array = dist_matrix[i].flatten()

            for key, focus in self.ac_dict.items():
                if focus == ac:
                    continue

                if dist_array[np.argwhere(ac_keys == focus.id)[0]] > 200:
                    continue


                ac.genRelativeState(focus, dist_array[np.argwhere(ac_keys == key)[0]])


            ac.step()

            for vol in ac.next_wpt.volumes:
                v = self.volume_dict.get_volume_by_id(vol)
                v.getNearestIntersection(ac.position,ac.heading)

        success_terminal, fail_terminal = self.checkTerminal(dist_matrix)

        self.success += len(success_terminal)
        self.fail += len(fail_terminal)

        for ac in success_terminal:
            self.ac_dict.remove(ac)
            self.active -= 1

        for ac in fail_terminal:
            self.ac_dict.remove(ac)
            self.active -= 1

        if self.visual: self.render()
        self.counter += 1

        return self.num_aircraft != self.max_aircraft or self.active != 0
    
    def genAircraft(self):

        for i, offset in enumerate(self.offsets):
            if self.num_aircraft >= self.max_aircraft:
                return

            if offset == self.counter:
                ac = Aircraft(self.num_aircraft, self.waypoint_dict.waypoint_dict[self.positions[i]].position,np.random.randint(1,6),self.routes[i].theta,self.counter, self.routes[i])
                self.ac_dict.add(ac)

                self.offsets[i] += np.random.choice(self.separations,1)[0]

                self.num_aircraft += 1
                self.active += 1

    def checkTerminal(self, dist_matrix):
        fail = set()
        success = set()

        ac_keys = np.array(list(self.ac_dict.ac_dict.keys())).flatten()

        for _, ac in self.ac_dict.items():
            i = np.argwhere(ac_keys == ac.id)[0]
            x = dist_matrix[i,:].flatten()
            x[i] = self.min_dist*10

            y = np.round(x-self.min_dist)

            conflicts = np.argwhere(y <=0).flatten()

            if len(conflicts) > 0:
                for conflict in conflicts:
                    fail.add(self.ac_dict.get_aircraft_by_id(ac_keys[conflict]))
                fail.add(ac)
                continue

            if ac.getTerminalSuccess():
                success.add(ac)
                continue

            ac_next = ac.next_wpt

            is_valid = any(
                self.volume_dict.get_volume_by_id(volume).getPointIn(ac.position)
                for volume in ac_next.volumes
            )

            if not is_valid:
                fail.add(ac)
        
        return success, fail


    def render(self):
        from colour import Color
        from gym.envs.classic_control import rendering

        red = Color('red')
        colors = list(red.range_to(Color('green'), self.max_aircraft))
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.viewport[0], self.viewport[1])
            self.viewer.set_bounds(0, self.viewport[0], 0, self.viewport[1])
            for _, volume in self.volume_dict.volume_dict.items():
                for i in range(len(volume.points)):
                    line = rendering.Line(volume.points[i-1], volume.points[i])
                    self.viewer.add_geom(line)
            
            for _, waypoint in self.waypoint_dict.waypoint_dict.items():
                wpt_img = rendering.Image(os.path.join(__location__, '../images/VOR.png'), 20, 20)
                jtransform = rendering.Transform(translation=waypoint.position)
                wpt_img.add_attr(jtransform)
                wpt_img.set_color(0, 0, 0)
                self.viewer.add_geom(wpt_img)


        for id, aircraft in self.ac_dict.ac_dict.items():
            aircraft_img = rendering.Image(os.path.join(__location__, '../images/drone.png'), 20, 20)
            jtransform = rendering.Transform(rotation=aircraft.heading - pi / 2, translation=aircraft.position)
            aircraft_img.add_attr(jtransform)
            r, g, b = colors[aircraft.id % self.num_aircraft].get_rgb()
            aircraft_img.set_color(r, g, b)
            self.viewer.onetime_geoms.append(aircraft_img)
        
        return self.viewer.render(return_rgb_array=False)
    
    def reset(self):
        self.num_aircraft = 0
        self.active = 0
        self.counter = 0
        self.success, self.fail = 0, 0

        self.offsets = np.random.choice(self.separations,len(self.positions))



class AircraftDict:
    # all of the aircraft object is stored in this class
    def __init__(self):
        self.ac_dict = OrderedDict()
    
    def items(self):
        return self.ac_dict.items()

    def genDistMatrix(self):
        positions = np.array([complex(ac.position[0],ac.position[1]) for _, ac in self.ac_dict.items()], dtype=np.complex64)

        m, n = np.meshgrid(positions, positions)
        dist_matrix = abs(m-n)
        dist_matrix = np.where(dist_matrix >= 200,np.inf,dist_matrix)
        return dist_matrix

    # how many aircraft currently en route
    @property
    def num_aircraft(self):
        return len(self.ac_dict)

    # add aircraft to dict
    def add(self, aircraft):
        # id should always be different
        assert aircraft.id not in self.ac_dict.keys(), 'aircraft id %d already in dict' % aircraft.id
        self.ac_dict[aircraft.id] = aircraft

    # remove aircraft from dict
    def remove(self, aircraft):
        with contextlib.suppress(KeyError):
            del self.ac_dict[aircraft.id]

    # get aircraft by its id
    def get_aircraft_by_id(self, aircraft_id):
        return self.ac_dict[aircraft_id]

class VolumeDict:
    def __init__(self):
        self.volume_dict = OrderedDict()
    
    # how many airspace volumes exist
    @property
    def num_volumes(self):
        return len(self.volume_dict)
    
    # add volume to dict
    def add(self, volume):
        # id should always be different
        assert volume.id not in self.volume_dict.keys(), 'volume id %d already in dict' % volume.id
        self.volume_dict[volume.id] = volume

        # remove volume from dict
    def remove(self, volume):
        with contextlib.suppress(KeyError):
            del self.volume_dict[volume.id]

    # get volume by its id
    def get_volume_by_id(self, volume_id):
        return self.volume_dict[volume_id]
    
    def items(self):
        return self.volume_dict.items()

class WaypointDict:
    def __init__(self):
        self.waypoint_dict = OrderedDict()
    
    # how many waypoints exist
    @property
    def num_volumes(self):
        return len(self.waypoint_dict)
    
    # add volume to dict
    def add(self, waypoint):
        # id should always be different
        assert waypoint.id not in self.waypoint_dict.keys(), 'waypoint id %d already in dict' % waypoint.id
        self.waypoint_dict[waypoint.id] = waypoint

        # remove volume from dict
    def remove(self, waypoint):
        with contextlib.suppress(KeyError):
            del self.waypoint_dict[waypoint.id]

    # get volume by its id
    def get_waypoint_by_id(self, waypoint_id):
        return self.waypoint_dict[waypoint_id]

    def items(self):
        return self.waypoint_dict.items()
    

