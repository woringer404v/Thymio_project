import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


class Print:
    '''
    Motion Control Module where all the other different modules are combined. 
    Check the report for a simple diagram where it is shown how it is used.
    '''
    def __init__(
        self,
        vision=None,
        globalnavi=None,
        controller=None,
        filtering=None,
        localnavi=None,
    ) -> None:
        self.img_to_print = None
        self.vision_ = vision
        self.globalnavi_ = globalnavi
        self.localnavi_ = localnavi
        self.controller_ = controller
        self.filtering_ = filtering
        self.scale_ = self.vision_.scale
        self.estimated_positions = []

    def plot_point(self, point, color="red", size=2):
        plt.plot(
            point[0],
            point[1],
            marker="o",
            markersize=size,
            markeredgecolor=color,
            markerfacecolor=color,
        )

    def plot_point_cv2(self, point, color=(0, 0, 0), radius=int(2)):
        cv2.circle(
            self.img_to_print,
            point,
            radius=radius,
            color=color,
            thickness=2,
        )

    def rescaled_point(self, point):
        return (round(point[0] / self.scale_), round(point[1] / self.scale_))

    def plot_path_cv2(self, path):
        for i in range(path.shape[1] - 1):
            cv2.line(
                self.img_to_print,
                self.rescaled_point((path[0, i], path[1, i])),
                self.rescaled_point((path[0, i + 1], path[1, i + 1])),
                (0, 0, 0),
                2,
            )
            self.plot_point_cv2(self.rescaled_point(path[:, i]))

    def print_init(self):
        self.img_to_print = self.vision_.planar_img
        self.plot_point(self.vision_.pos_thymio)
        self.plot_point(self.vision_.pos_blue)
        self.plot_point(self.vision_.goal)
        plt.imshow(self.img_to_print)

    def print(self):
        self.img_to_print = self.vision_.planar_img
        path = self.globalnavi_.path

        # Print thymio's trajectory
        for i, p in enumerate(self.vision_.positions):
            self.plot_point_cv2(p, (255, 0, 0))
            self.plot_point_cv2(self.vision_.green_positions[i], (0, 255, 0))

        # Print goal
        self.plot_point_cv2(self.vision_.goal, (0, 0, 255))

        # Print local goal
        rescaled_goal = self.rescaled_point(self.controller_.local_goal)
        local_goal = (int(rescaled_goal[0]), int(rescaled_goal[1]))
        self.plot_point_cv2(local_goal, (0, 0, 255), 6)

        # Rescale and print the global path
        self.plot_path_cv2(path)

        # Print the filter estimates
        # Need to save an array of the estimates somewhere, pass it to print. Print will show all the estimates on each new image.
        previous_pos = self.rescaled_point(
            np.array(self.filtering_.previous_position_[:2])
        )
        self.estimated_positions.append(previous_pos)
        for pos in self.estimated_positions:
            self.plot_point_cv2(pos, (0, 0, 255))

        cv2.imshow("frame", self.img_to_print)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return 0

    def plot_path(self):
        _, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("Path Planning with A*", fontsize=16)
        major_ticks_x = np.arange(0, self.globalnavi_.maze.shape[0] + 1, 5)
        minor_ticks_x = np.arange(0, self.globalnavi_.maze.shape[0] + 1, 1)
        major_ticks_y = np.arange(0, self.globalnavi_.maze.shape[1] + 1, 5)
        minor_ticks_y = np.arange(0, self.globalnavi_.maze.shape[1] + 1, 1)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.grid(which="minor", alpha=0.2)
        ax.grid(which="major", alpha=0.5)
        ax.set_xlim([-1, self.globalnavi_.maze.shape[0]])
        ax.set_ylim([self.globalnavi_.maze.shape[1], -1])
        ax.grid(True)
        ax.arrow(
            self.globalnavi_.start[0],
            self.globalnavi_.start[1],
            2 * math.cos(self.globalnavi_.orient * np.pi / 180),
            2 * math.sin(self.globalnavi_.orient * np.pi / 180),
            color="b",
            width=0.01,
        )
        ax.imshow(
            np.transpose(1 - self.globalnavi_.maze),
            cmap="gray",
            interpolation="nearest",
            alpha=0.2,
            origin="upper",
        )
        ax.imshow(
            np.transpose((1 - self.globalnavi_.initial_maze)),
            interpolation="nearest",
            cmap="gray",
            alpha=0.5,
            origin="upper",
        )
        ax.plot(
            self.globalnavi_.path[0],
            self.globalnavi_.path[1],
            marker="o",
            color="black",
        )
        ax.scatter(
            self.globalnavi_.start[0],
            self.globalnavi_.start[1],
            marker="o",
            color="b",
            s=200,
        )
        ax.scatter(
            self.globalnavi_.goal[0],
            self.globalnavi_.goal[1],
            marker="o",
            color="r",
            s=200,
        )

    def plot_distances_map(self):
        _, ax = plt.subplots(figsize=(7, 7))
        ax.set_title("Distance map for Local Avoidance", fontsize=16)
        major_ticks_x = np.arange(0, self.localnavi_.distances_table.shape[0] + 1, 5)
        minor_ticks_x = np.arange(0, self.localnavi_.distances_table.shape[0] + 1, 1)
        major_ticks_y = np.arange(0, self.localnavi_.distances_table.shape[1] + 1, 5)
        minor_ticks_y = np.arange(0, self.localnavi_.distances_table.shape[1] + 1, 1)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.grid(which="minor", alpha=0.2)
        ax.grid(which="major", alpha=0.5)
        ax.set_xlim([-1, self.localnavi_.distances_table.shape[0]])
        ax.set_ylim([self.localnavi_.distances_table.shape[1], -1])
        ax.grid(True)
        ax.imshow(
            np.transpose(self.localnavi_.distances_table),
            cmap="gray",
            interpolation="nearest",
            origin="upper",
        )
