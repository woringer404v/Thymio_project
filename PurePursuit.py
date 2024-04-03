"""Source code : https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit"""


import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

LOOKAHEADIS = 2
KP = 3


class PurePursuit:
    def __init__(
        self,
        path,
        initialPos,
        initialOrientation,
    ) -> None:
        self.path = path
        self.initialPos = initialPos
        self.initialOrientation = initialOrientation  # in degree
        self.lastFoundIndex = 0

    def update_turnVel(self, currentPos, currentHeading) -> float:
        currentX, currentY = currentPos
        startingIndex = self.lastFoundIndex

        translated_path = self.path - currentPos
        for i in range(startingIndex, len(self.path) - 1):

            # beginning of line-circle intersection code
            p1 = translated_path[i]
            p2 = translated_path[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dr = math.dist(p1, p2)
            D = np.hstack([np.reshape(p1, (2, 1)), np.reshape(p2, (2, 1))])
            discriminant = (LOOKAHEADIS**2) * (dr**2) - D**2

            if discriminant >= 0:
                sol_x1 = (D * dy + np.sign(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y1 = (-D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2

                sol_x2 = (D * dy - np.sign(dy) * dx * np.sqrt(discriminant)) / dr**2
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
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2

                    # if not both solutions are in range, take the one that's in range
                    else:
                        # if solution pt1 is in range, set that as goal point
                        if (minX <= sol_pt1[0] <= maxX) and (
                            minY <= sol_pt1[1] <= maxY
                        ):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if math.dist(goalPt, self.path[i + 1]) < math.dist(
                        [currentX, currentY], self.path[i + 1]
                    ):
                        # update lastFoundIndex and exit
                        lastFoundIndex = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        lastFoundIndex = i + 1

                # if no solutions are in range
                else:
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    goalPt = [
                        self.path[lastFoundIndex][0],
                        self.path[lastFoundIndex][1],
                    ]

        # obtained goal point, now compute turn vel
        # initialize proportional controller constant

        # calculate absTargetAngle with the atan2 function
        absTargetAngle = (
            math.atan2(goalPt[1] - currentPos[1], goalPt[0] - currentPos[0])
            * 180
            / np.pi
        )
        if absTargetAngle < 0:
            absTargetAngle += 360

        # compute turn error by finding the minimum angle
        turnError = absTargetAngle - currentHeading
        if turnError > 180 or turnError < -180:
            turnError = -1 * np.sign(turnError) * (360 - abs(turnError))

        # apply proportional controller
        turnVel = KP * turnError

        return turnVel
