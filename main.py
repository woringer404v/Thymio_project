import Vision, GlobalNavigation, Filtering, LocalNavigation, Control, MotionControl, Robot
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# todo: make sure that the positions predicted by camera, measurement and the one used in the path are the same! (camera is the green point, measurement is between wheels, path uses center of object)
def main():

    # initial state
    vision = Vision.Vision()  # start connection with camera
    (
        maze,
        pos_thymio,
        angle_thymio,
        goal,
    ) = (
        vision.initialize_maze()
    )  # outputs maze (grid), (x,y) of thymio in grid coord., angle in degrees with horizontal, (x,y) of goal in grid coord.
    global_navigation = GlobalNavigation.GlobalNavigation(
        1 - maze,
        angle_thymio,
        np.around(pos_thymio),
        np.around(goal),
    )
    plt.plot(maze, "gray")
    print(np.around(pos_thymio))
    print(np.around(goal))
    path = global_navigation.A_Star()  # maybe the connectivity ?

    robot = Robot(vision.get_scale_grid_to_cm())  # start connection with robot
    filter = Filtering.create_filter(robot.get_wheel_distance())
    local_navigation = LocalNavigation(path)  # map?
    controller = Control.PurePursuit(path)
    motion_control = MotionControl(
        robot, vision, filter, local_navigation, controller
    )  # map needed?

    # enter the loop states
    motion_control.follow_path()

    vision.close_camera()  # close the camera

    return


if __name__ == "__main__":
    main()
