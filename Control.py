import numpy as np
import math

LOOKAHEADIS = 8
KP = 2
FORWARDSPEED = 100
CLOOSE_GOAL = 5


def sgn(num):
    if num >= 0:
        return 1
    else:
        return -1


class PurePursuit:
    '''
    Control Module that implements the Pure Pursuit Method.
    More information in the report.
    '''
    def __init__(
        self,
        path,
    ) -> None:
        self.path = np.transpose(path)
        self.lastFoundIndex = 0
        self.local_goal = self.path[0]

    def update_local_goal(self, currentPos):
        currentX, currentY = currentPos

        for i in range(self.lastFoundIndex, len(self.path) - 1):

            # beginning of line-circle intersection code
            x1 = self.path[i][0] - currentX
            y1 = self.path[i][1] - currentY
            x2 = self.path[i + 1][0] - currentX
            y2 = self.path[i + 1][1] - currentY
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt(dx**2 + dy**2)
            D = x1 * y2 - x2 * y1
            discriminant = (LOOKAHEADIS**2) * (dr**2) - D**2

            if discriminant >= 0:
                sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y1 = (-D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2

                sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y2 = (-D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

                sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
                sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
                # end of line-circle intersection code

                minX = min(self.path[i][0], self.path[i + 1][0])
                minY = min(self.path[i][1], self.path[i + 1][1])
                maxX = max(self.path[i][0], self.path[i + 1][0])
                maxY = max(self.path[i][1], self.path[i + 1][1])

                # if one or both of the solutions are in range
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or (
                    (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)
                ):

                    # if both solutions are in range, check which one is better
                    if (
                        (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)
                    ) and (
                        (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)
                    ):
                        # make the decision by compare the distance between the intersections and the next point in path
                        if math.dist(sol_pt1, self.path[i + 1]) < math.dist(
                            sol_pt2, self.path[i + 1]
                        ):
                            self.local_goal = sol_pt1
                        else:
                            self.local_goal = sol_pt2

                    # if not both solutions are in range, take the one that's in range
                    else:
                        # if solution pt1 is in range, set that as goal point
                        if (minX <= sol_pt1[0] <= maxX) and (
                            minY <= sol_pt1[1] <= maxY
                        ):
                            self.local_goal = sol_pt1
                        else:
                            self.local_goal = sol_pt2

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if math.dist(self.local_goal, self.path[i + 1]) < math.dist(
                        [currentX, currentY], self.path[i + 1]
                    ):
                        # update lastFoundIndex and exit
                        self.lastFoundIndex = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        self.lastFoundIndex = i + 1

                # if no solutions are in range
                else:
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    self.local_goal = [
                        self.path[self.lastFoundIndex][0],
                        self.path[self.lastFoundIndex][1],
                    ]
            if math.dist(self.local_goal, currentPos) < CLOOSE_GOAL:
                self.local_goal = self.path[-1]

        # obtained goal point, now compute turn vel
        # initialize proportional controller constant

        # calculate absTargetAngle with the atan2 function

    def calculate_turnVel(self, currentPos, currentHeading):
        absTargetAngle = (
            np.arctan2(
                (self.local_goal[1] - currentPos[1]),
                self.local_goal[0] - currentPos[0],
            )
            * 180
            / np.pi
        )
        if absTargetAngle < 0:
            absTargetAngle += 360

        turnError = absTargetAngle - currentHeading
        turnError = (turnError + 180) % 360 - 180

        turnVel = KP * turnError
        return turnVel

    def iterate(self, position, angle):
        self.update_local_goal(position)
        turnVel = self.calculate_turnVel(position, angle)
        return [round(FORWARDSPEED + turnVel), round(FORWARDSPEED - turnVel)]

    def orientation(
        self,
        parent,
        current,
    ):
        return np.array([current[0] - parent[0], current[1] - parent[1]])
