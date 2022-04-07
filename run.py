from src.environment import Environment

env = Environment(max_aircraft=2, visual= True)
epochs = 1
for i in range(epochs):
    running = True

    while running:
        running = env.step()


    if (i+1) % 20 == 0:
        print(f"EPOCH {i+1}")
        print(env.success, env.fail)
        print('===============================')

    env.reset()
