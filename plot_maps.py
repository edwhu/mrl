import numpy as np
import matplotlib.pyplot as plt
from envs.sibrivalry.toy_maze import PointMaze2D

def plot_heatmap(x_min, x_max, y_min, y_max, x_div, y_div, goal_x, goal_y):
    if goal_x < x_min or goal_x > x_max:
        raise ValueError("invalid goal x coord")
    if goal_y < y_min or goal_y > y_max:
        raise ValueError("invalid goal y coord")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("invalid coordinate ranges")
    x = np.linspace(x_min, x_max, x_div)
    y = np.linspace(y_min, y_max, y_div)
    XY = X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)
    XY = XY.reshape(x_div * y_div, 2)
    goal_vec = np.zeros((x_div*y_div, 2))
    goal_vec[:,0] = goal_vec[:,0] + goal_x
    goal_vec[:,1] = goal_vec[:,1] + goal_y
    XY_goal = np.hstack((XY, goal_vec))
    # need learned reward
    placeholder_output =  np.random.random((x_div * y_div,))
    plt.tricontourf(XY[:, 0], XY[:, 1], placeholder_output, z = 1)
    plt.colorbar()
    plt.title("Output of learned reward function")
    plt.savefig('heatmap.png', dpi='figure', bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    maze = PointMaze2D()
    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    maze.maze.plot(ax) # plot the walls
    plot_heatmap(-0.5, 9.5, -0.5, 9.5, 100, 100, maze.g_xy[0], maze.g_xy[1])
    