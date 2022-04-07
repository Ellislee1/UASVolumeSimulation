from numpy import radians

G = 9.8
scale = 1
min_speed = 1 / scale
max_speed = 5 / scale
speed_sigma = 0.5 / scale     # Speed uncertainty
d_heading = radians(0)
heading_sigma = radians(1)  # heading uncertainty