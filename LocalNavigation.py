import numpy as np
from typing import Tuple, List

TOO_CLOSE_DIST = 10
TOO_CLOSE_DIST_STUCK = 6
THRESHOLD_VALUE = 3
FORWARD_SPEED = 100
LEFT = 0
RIGHT = 1


class LocalNavigation(object):
    '''
    Module that implements the local navigation part of the system.
    '''
    def __init__(self, path, distances_table):
        self.path = np.transpose(path)
        self.distances_table = distances_table
        self.obstacle_on_side = 2
        self.obstacle_on_side_white = 2
        self.chosen_direct = None
        self.stuck = 0
        self.movements = "8N"

    def _orientation_4n(self):
        return [
            ((0, -1), (0, 1)),
            ((1, 0), (-1, 0)),
            ((0, 1), (0, -1)),
            ((-1, 0), (1, 0)),
        ]

    def _orientation_8n(self):
        return [
            ((0, -1), (0, 1)),
            ((1, 0), (0, 1)),
            ((1, 0), (-1, 0)),
            ((0, 1), (-1, 0)),
            ((0, 1), (0, -1)),
            ((-1, 0), (0, -1)),
            ((-1, 0), (1, 0)),
            ((0, -1), (1, 0)),
        ]

    def discret_init_orientation(self, angle):
        idx = round(angle / 45.0) % 8
        return idx

    def get_orientation(
        self,
    ) -> List[Tuple[int, int, float]]:
        if self.movements == "4N":
            return self._orientation_4n()
        elif self.movements == "8N":
            return self._orientation_8n()
        else:
            raise ValueError("Unknown movement")

    def obstacle_detection(self, prox_horizontal):
        prox_sum = 0

        for i in range(5):
            prox_sum = prox_sum + prox_horizontal[i]

        return prox_sum > 200

    def is_black_too_close(self, position):
        position_table_thymio = (round(position[0]), round(position[1]))
        return self.distances_table[position_table_thymio] < TOO_CLOSE_DIST

    def is_black_too_close_stuck(self, position):
        position_table_thymio = (round(position[0]), round(position[1]))
        return self.distances_table[position_table_thymio] < TOO_CLOSE_DIST_STUCK

    def prioritize_side(self, position):
        position_table_thymio = (round(position[0]), round(position[1]))

        orientation_left = self.get_orientation()[
            self.discret_init_orientation(position[2])
        ][LEFT]
        cell_left = (
            position_table_thymio[0] + orientation_left[0],
            position_table_thymio[1] + orientation_left[1],
        )

        orientation_right = self.get_orientation()[
            self.discret_init_orientation(position[2])
        ][RIGHT]
        cell_right = (
            position_table_thymio[0] + orientation_right[0],
            position_table_thymio[1] + orientation_right[1],
        )

        if (
            self.is_black_too_close_stuck(cell_left)
            and self.is_black_too_close_stuck(cell_right)
        ) and self.distances_table[position_table_thymio] >= np.max(
            (self.distances_table[cell_right], self.distances_table[cell_left])
        ):
            self.stuck = 1
        elif self.distances_table[cell_right] >= self.distances_table[cell_left]:

            self.chosen_direct = RIGHT
            self.obstacle_on_side = LEFT
        elif self.distances_table[cell_right] < self.distances_table[cell_left]:
            self.chosen_direct = LEFT
            self.obstacle_on_side = RIGHT

    def distance_to_path(self, position):
        path = self.path
        position = np.asarray(position)
        dist_2 = np.sum((path - position) ** 2, axis=1)
        min1_idx, min2_idx = np.argpartition(dist_2, 1)[0:2]
        area = np.cross(position - path[min1_idx, :], position - path[min2_idx, :])

        diagonal = np.linalg.norm(path[min1_idx, :] - path[min2_idx, :], 2)
        distance = area / diagonal
        distance = abs(distance)
        return distance

    def is_back_to_path(self, position):
        distance = self.distance_to_path(position)
        return distance < THRESHOLD_VALUE

    def turn_around(self):
        y = [0, 0]
        ext_speed = 150
        int_speed = 50

        if self.obstacle_on_side == RIGHT:
            y[0] = ext_speed
            y[1] = int_speed
        elif self.obstacle_on_side == LEFT:
            y[0] = int_speed
            y[1] = ext_speed

        return y

    def avoid_obstacle(self, prox_horizontal, position):
        sensor_scale = 1500
        y = [0, 0]
        w_l = [60, 50, -25, -50, -60]
        w_r = [-60, -50, 25, 50, 60]
        x = [0, 0, 0, 0, 0]

        black_on_side = self.is_black_too_close(position)  

        for i in range(5):
            x[i] = prox_horizontal[i] // sensor_scale
            y[0] = y[0] + x[i] * w_l[i]
            y[1] = y[1] + x[i] * w_r[i]

        if not black_on_side: 

            y[0] = FORWARD_SPEED + y[0]
            y[1] = FORWARD_SPEED + y[1]
            if y[0] > y[1]:
                self.obstacle_on_side = LEFT
            elif y[0] < y[1]:
                self.obstacle_on_side = RIGHT

        else: 
            if self.chosen_direct == None:
                self.prioritize_side(position)
            if not self.stuck:
                if self.chosen_direct == LEFT:
                    y = [-100, 100]
                else:
                    y = [100, -100]
            else:
                return False

        return y

    def iterate(self, prox_sensors, position):
        if self.obstacle_detection(prox_sensors):
            return self.avoid_obstacle(prox_sensors, position)
        else:
            return self.turn_around()
