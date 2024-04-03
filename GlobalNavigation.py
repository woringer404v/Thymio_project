import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure
import heapq


def dilate(initial_maze, dilation, struct):
    """
    Dilation function used to evade putting the thymio over the non-traversable box.

    Args:
        initial_maze (np.array): the maze whose obstacles need to be dilated
        dilation (int): the radius of the dilation to be applied
        struct (np.array): the type of structure to be used in the dilation

    Returns:
        np.array: the dilated maze, of the same shape as initial_maze but
                  with dilated obstacles
    """
    if dilation == 0:
        return initial_maze
    struct = generate_binary_structure(2, struct)
    pad_maze = np.pad(initial_maze, (1,), constant_values=(1,))
    return binary_dilation(pad_maze, structure=struct, iterations=dilation)[1:-1, 1:-1]


class GlobalNavigation:
    """
    Global Navigation Module.
    It is used to get the nominal trajectory in the created maze, by using A-Star.

    The implementation takes as state the coordinates (x,y,theta) and optimizes for
    the path length, being the path length the sum of all edges' length of the path,
    where the length is computed as a linear combination of the spatial distance and
    the angle distance between the subsequent states. For more info, check the report.

    There is a dilation applied on the given maze to make the obstacles bigger
    so that the Thymio doesn't go over these obstacles.
    """

    def __init__(
        self,
        maze,
        initial_orientation,
        start,
        goal,
        dilation,
        movements: str = "8N",
        alpha=1.0,
    ) -> None:
        """
        Args:
            maze (np.array): maze where the A-star needs to be run on after dilation.
            initial_orientation (int): initial orientation of thymio in degrees
            start ((int,int)): start position of thymio in grid units
            goal ((int,int)): goal position in grid units
            dilation (int): radius of structure in dilation
            movements (str, optional): Type of connectivity in graph, either "4N" (4-connectivity, only vertical and horizontal movements) or "8N" (8-connectivity, "4N" with also diagonal movements). Defaults to "8N".
            alpha (float, optional): linear combination coefficient for edge length. Defaults to 1.
        """
        self.initial_maze = maze
        self.orient = initial_orientation
        self.movements = movements
        self.start = start
        self.goal = goal
        self.dilation = dilation
        self.maze = dilate(maze, int(dilation), 2)
        self.path = None
        self.DISTANCE_STOP = 3
        self.alpha = alpha

    def discret_init_orientation(self):
        """
        Discretizes the initial orientation of [0,360) to [0,8) (by approximating
        to the closest 45 degree multiple).

        Returns:
            int: [0,8) index state of angle
        """
        idx_min = round((self.orient) / 45.0) % 8
        return idx_min

    def at_goal(self, currentPosition):
        """
        Checks if position is the goal.
        Returns:
            bool: is at goal
        """
        return math.dist(currentPosition, self.goal) < self.DISTANCE_STOP

    def reconstruct_path(self, cameFrom, current):
        total_path = [current[:2]]

        while current[:2] != self.start:
            total_path.insert(0, cameFrom[current][:2])
            current = cameFrom[current]

        self.path = np.transpose(np.array(total_path).reshape(-1, 2))

        return self.path

    def _get_movements_4n(self):
        return [
            (1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
        ]

    def _get_movements_8n(self):
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

    def get_movements(self):
        """
        Gets different movements according to what was declared in the initialization.
        """
        if self.movements == "4N":
            return self._get_movements_4n()
        elif self.movements == "8N":
            return self._get_movements_8n()
        else:
            raise ValueError("Unknown movement")

    def orientation(self, dx, dy):
        """Maps movement to its resulting angle state.

        Args:
            dx (int): Increment in x in movement
            dy (int): Increment in y in movement

        Returns:
            int: angle state in [0,8)
        """
        orientation_states = {
            (1, 0): 0,
            (1, 1): 1,
            (0, 1): 2,
            (-1, 1): 3,
            (-1, 0): 4,
            (-1, -1): 5,
            (0, -1): 6,
            (1, -1): 7,
        }

        return orientation_states[(dx, dy)]

    def turn_cost(self, current, neighbor):
        """
        Computes the turning cost (weigthed difference in angles) between two given states.

        Args:
            current ((int,int,int)): Given first state
            neighbor ((int,int,int)): Given second state

        Returns:
            (float): The cost of changing orientation.
        """
        current_orientation = current[2]
        next_orientation = neighbor[2]

        orientation_diff = (next_orientation - current_orientation + 4) % 8 - 4

        return self.alpha * abs(orientation_diff)

    def A_Star(self):
        """
        A-Star algorithm, which uses a heapq as a priority queue.

        For the heuristic, it's using the grid distance.

        Raises:
            Exception: Start Node not Traversable
            Exception: Goal Node not Traversable

        Returns:
            list: Path result
        """

        for point in [self.start, self.goal]:
            assert (
                0 <= point[0] <= self.maze.shape[0]
                and 0 <= point[1] <= self.maze.shape[1]
            ), "start or end goal not contained in the map"
        if self.maze[self.start[0], self.start[1]] == 1:
            raise Exception("Start node is not traversable")

        if self.maze[self.goal[0], self.goal[1]] == 1:
            raise Exception("Goal node is not traversable")

        start_node = (self.start[0], self.start[1], self.discret_init_orientation())

        gScore = np.ones([self.maze.shape[0], self.maze.shape[1], 8]) * np.inf
        gScore[start_node] = 0

        # This is grid distance (more accurate than euclidean)
        hScore = np.array(
            [
                [
                    np.sqrt(2)
                    * np.min((abs(row - self.goal[0]), abs(col - self.goal[1])))
                    + abs((abs(row - self.goal[0]) - abs(col - self.goal[1])))
                    for col in range(self.maze.shape[1])
                ]
                for row in range(self.maze.shape[0])
            ]
        )

        openList = [(hScore[self.start], hScore[self.start], 0, start_node)]
        closedSet = set()
        cameFrom = {start_node: "Start"}

        heapq.heapify(openList)
        while len(openList) != 0:

            top = heapq.heappop(openList)
            current = top[-1]

            if current[:2] == self.goal:
                return self.reconstruct_path(cameFrom, current)

            if current in closedSet:
                continue
            closedSet.add(current)

            for dx, dy, deltacost in self.get_movements():

                # state is x, y, angle
                neighbor = (current[0] + dx, current[1] + dy, self.orientation(dx, dy))

                # if the neighbor is not in the map or is occupied, skip
                if (
                    (neighbor[0] >= self.maze.shape[0])
                    or (neighbor[1] >= self.maze.shape[1])
                    or (neighbor[0] < 0)
                    or (neighbor[1] < 0)
                    or self.maze[neighbor[0], neighbor[1]]
                ):
                    continue

                # if the state has already been visited (only works
                # with consistent heuristic), skip
                if neighbor in closedSet:
                    continue

                tentative_gScore = (
                    gScore[current] + deltacost + self.turn_cost(current, neighbor)
                )

                # heuristic just depends on neighbor position
                heur = hScore[neighbor[:2]]

                if gScore[neighbor] > tentative_gScore:

                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    heapq.heappush(
                        openList,
                        (tentative_gScore + heur, heur, tentative_gScore, neighbor),
                    )

        print("No path found to goal")
        return []

    def distance_map(self):
        """
        Precomputation of distance from every point to its closest object.

        Returns:
            np.array: array of smallest distance to object
        """
        distance_map = 1 * self.maze.copy()
        while 0 in distance_map:
            distance_map += 1 * dilate(distance_map, 1, 1)
        return np.max(distance_map) - distance_map
