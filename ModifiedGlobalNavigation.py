from typing import Tuple, List, List, Dict
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure
import heapq
import math

discret_angle: List = [0, 90, 180, 270, 45, 135, 225, 315]
DISTANCE_STOP = 1

# 12 18 
# 59 24

class GlobalNavigation:
    def __init__(
        self,
        maze,
        initial_orientation,
        start,
        goal,
        dilation,
        movements: str = "8N",
    ) -> None:
        self.maze = maze
        self.orient = initial_orientation
        self.movements = movements
        self.start = start
        self.goal = goal
        self.dilation = dilation

    def discret_init_orientation(self):
        idx_min = round((self.orient)/45.)%8
        cameFrom = self.get_movements()[idx_min][:2]
        return cameFrom

    def reconstruct_path(
        self,
        cameFrom,
        current,
    ):
        print("Path is being reconstructed")
        total_path = [current]
        while current != self.start:
            total_path.insert(0, cameFrom[current])
            # old_current = current
            current = cameFrom[current]
            # del cameFrom[old_current]
        self.path = np.transpose(np.array(total_path).reshape(-1, 2))

        print("Path computed")
        return self.path

    def _get_movements_4n(self) -> List[Tuple[int, int, float]]:
        return [
            (1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
        ]

    def _get_movements_8n(self) -> List[Tuple[int, int, float]]:
        s2 = math.sqrt(2)
        return [
            (1, 0, 1.0),
            (1, 1, s2),
            (0, 1, 1.0),
            (-1, 1, s2),
            (-1, 0, 1.0),
            (-1, -1, s2),
            (0, -1, 1.0),
            (1, -1, s2),
        ]

    def get_movements(
        self,
    ) -> List[Tuple[int, int, float]]:
        if self.movements == "4N":
            return self._get_movements_4n()
        elif self.movements == "8N":
            return self._get_movements_8n()
        else:
            raise ValueError("Unknown movement")

    def orientation(
        self,
        parent,
        current,
    ):
        return (current[0] - parent[0], current[1] - parent[1])

    def turn_cost(
        self,
        parent,
        current,
        neighbor,
        alpha=1,
    ):
        current_orientation = self.orientation(parent, current)
        next_orientation = self.orientation(current, neighbor)
        turn = abs((current_orientation[0] - next_orientation[0])) + abs(
            (current_orientation[1] - next_orientation[1])
        )

        return alpha * turn

    def A_Star(self):
        self.dilate()
        self.maze[self.start] = 0.5
        self.maze[self.start] = 0
        for point in [self.start, self.goal]:
            assert (
                0 <= point[0] <= self.maze.shape[0]
                and 0 <= point[1] <= self.maze.shape[1]
            ), "start or end goal not contained in the map"
        if self.maze[self.start[0], self.start[1]] == 1:
            raise Exception("Start node is not traversable")

        if self.maze[self.goal[0], self.goal[1]] == 1:
            raise Exception("Goal node is not traversable")

        openList = [self.start]
        closedList = []
        cameFrom = {
            self.start: (
                self.start[0] - self.discret_init_orientation()[0],
                self.start[1] - self.discret_init_orientation()[1],
            )
        }
        gScore = np.ones(self.maze.shape) * np.inf
        gScore[self.start] = 0

        # This is grid distance (more accurate than euclidean)
        hScore = np.array(
            [
                [
                    math.dist((row, col), self.goal)
                    # np.sqrt(2)*np.min(abs(row - self.goal[0]), abs(col - self.goal[1])) + abs((row - self.goal[0] - (col - self.goal[1])))
                    for col in range(self.maze.shape[1])
                ]
                for row in range(self.maze.shape[0])
            ]
        )

        # fScore = np.ones(self.maze.shape) * np.inf
        # fScore[self.start] = gScore[self.start] + hScore[self.start]

        def get_fScore(position) -> float:
            return gScore[position] + hScore[position]

        # might need to put another variable to the state: the angle (8 dif ones) 
        # and the cost to move between them
        while openList:

            current = min(openList, key=get_fScore)

            if current == self.goal:
                print("Goal found")
                return self.reconstruct_path(cameFrom, current)

            openList.remove(current)
            closedList.append(current)

            for dx, dy, deltacost in self.get_movements():

                neighbor = (current[0] + dx, current[1] + dy)

                # out of bounds
                if (
                    (neighbor[0] >= self.maze.shape[0])
                    or (neighbor[1] >= self.maze.shape[1])
                    or (neighbor[0] < 0)
                    or (neighbor[1] < 0)
                ):
                    continue
                
                # if it is a wall grid position
                # or it has already been explored (this only works if the heuristic 
                # is consistent and if the states are well defined!)
                if (self.maze[neighbor[0], neighbor[1]]) or (neighbor in closedList):
                    continue

                curr_gScore = (
                        gScore[current]
                        + deltacost
                        + self.turn_cost(cameFrom[current], current, neighbor)
                    )
                
                # would be better to have a set as the openlist (ordered? since there 
                # are a lot of additions)
                if gScore[neighbor] > curr_gScore:
                    # what about the cases that the found path is better than 
                    # the previous minimum?
                    if np.isinf(gScore[neighbor]):
                        openList.append(neighbor)
                    cameFrom[neighbor] = current
                    gScore[neighbor] = curr_gScore

        print("No path found to goal")
        return []

    def dilate(self):
        struct = generate_binary_structure(2, 2)
        self.maze = binary_dilation(
            self.maze, structure=struct, iterations=self.dilation
        )

    def at_goal(self, currentPosition) -> bool:
        print(f"Current Position {currentPosition}")
        print(f"Goal {self.goal}")
        return math.dist(currentPosition, self.goal) < DISTANCE_STOP

    def plot(self):
        _, ax = plt.subplots(figsize=(7, 7))

        major_ticks_x = np.arange(0, self.maze.shape[0] + 1, 5)
        minor_ticks_x = np.arange(0, self.maze.shape[0] + 1, 1)
        major_ticks_y = np.arange(0, self.maze.shape[1] + 1, 5)
        minor_ticks_y = np.arange(0, self.maze.shape[1] + 1, 1)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.grid(which="minor", alpha=0.2)
        ax.grid(which="major", alpha=0.5)
        ax.set_xlim([-1, self.maze.shape[0]])
        ax.set_ylim([-1, self.maze.shape[1]])
        ax.grid(True)

        ax.imshow(
            np.transpose(1 - self.maze),
            cmap="gray",
            interpolation="nearest",
            alpha=0.2,
            origin="lower",
        )
        ax.imshow(
            np.transpose((1 - self.maze)),
            interpolation="nearest",
            cmap="gray",
            alpha=0.5,
            origin="lower",
        )
        ax.plot(self.path[0], self.path[1], marker="o", color="black")
        ax.scatter(self.start[0], self.start[1], marker="o", color="b", s=200)
        ax.scatter(self.goal[0], self.goal[1], marker="o", color="r", s=200)
