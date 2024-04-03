import Print
import numpy as np
import Vision, GlobalNavigation
from ipywidgets import interact, IntSlider, ToggleButton, Layout


def interact_global_navigation():
    style = {"description_width": "initial"}

    def change_alpha_dil(alpha, movements, angle):
        if movements == False:
            movements = "8N"
        else:
            movements = "4N"
        if alpha == True:
            aplha = 1
        else:
            aplha = 0
        np.random.seed(seed=10)
        maze = np.random.randint(2, size=(50, 30))
        for i in range(6):
            maze = maze * np.random.randint(2, size=(50, 30))
        maze = GlobalNavigation.dilate(maze, 2, 2)
        maze0 = np.argwhere(maze == 0)
        goal = np.squeeze(
            maze0[np.random.choice(maze0.shape[0], 1, replace=False), :]
        )  # (15,12)
        start = np.squeeze(
            maze0[np.random.choice(maze0.shape[0], 1, replace=False), :]
        )  #
        gb = GlobalNavigation.GlobalNavigation(
            maze,
            angle,
            (start[0], start[1]),
            (goal[0], goal[1]),
            0,
            movements,
            alpha=alpha,
        )
        gb.A_Star()
        vis = Vision.Vision()

        prt = Print.Print(vis, globalnavi=gb)
        prt.plot_path()

    movements = ToggleButton(value=False, description="4N")
    alpha = ToggleButton(value=True, description="Penalize for turn")
    angle = IntSlider(
        min=0,
        max=315,
        step=45,
        value=0,
        description="Initial orientation",
        layout=Layout(width="98%"),
        style=style,
    )

    interact(
        change_alpha_dil,
        movements=movements,
        alpha=alpha,
        angle=angle,
    )
