from numpy import radians

G = 9.8
scale = 1
min_speed = 1 / scale
max_speed = 5 / scale
speed_sigma = 0.5 / scale     # Speed uncertainty
d_heading = radians(0)
heading_sigma = radians(1)  # heading uncertainty

simulate_frame = 10

window_width = 800
window_height = 800

min_sep = 15
goal_radius = 25