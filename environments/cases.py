import numpy as np
from src.volume import Volume, Waypoint, Route


def setupCase1(volume_dict, waypoint_dict):
    volume_dict.add(Volume(1, np.array([[75,375],[75,425],[725,425],[725,375]])))

    waypoint_dict.add(Waypoint("WPT1",np.array([100,400])))
    waypoint_dict.add(Waypoint("WPT2",np.array([700,400])))

    r1 = Route(0,[waypoint_dict.get_waypoint_by_id("WPT1"),waypoint_dict.get_waypoint_by_id("WPT2")])
    r2 = Route(1,[waypoint_dict.get_waypoint_by_id("WPT2"),waypoint_dict.get_waypoint_by_id("WPT1")])

    return volume_dict, waypoint_dict, [r1,r2]

def setupCase2(volume_dict, waypoint_dict):
    volume_dict.add(Volume(1, np.array([[175,225],[175,175],[625,175],[625,225]])))
    volume_dict.add(Volume(2, np.array([[575,175],[625,175],[625,625],[575,625]])))
    volume_dict.add(Volume(3, np.array([[165,200],[200,165],[635,600],[600,635]])))

    waypoint_dict.add(Waypoint("WPT1",np.array([200,200])))
    waypoint_dict.add(Waypoint("WPT2",np.array([600,200])))
    waypoint_dict.add(Waypoint("WPT3",np.array([600,600])))

    return volume_dict, waypoint_dict