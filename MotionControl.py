import time
import numpy as np
import Print as Pt


class MotionControl:
    '''
    Motion Control Module where all the other different modules are combined. 
    Check the report for a simple diagram where it is shown how it is used.
    '''
    def __init__(
        self,
        robot,
        vision,
        filter,
        local_navigation,
        global_navigation,
        controller,
    ):
        # modules
        self.robot_ = robot
        self.vision_ = vision
        self.filter_ = filter
        self.local_navigation_ = local_navigation
        self.global_navigation_ = global_navigation
        self.controller_ = controller

        # local vars
        self.in_local_navi_ = False
        self.time_local_navi = -1
        self.pt = Pt.Print(
            self.vision_, self.global_navigation_, self.controller_, self.filter_
        )
        self.previous_u = [0, 0]

    def local_navigation_is_needed(self, position, sensors, curr_time):
        if self.local_navigation_.obstacle_detection(sensors):
            self.time_local_navi = curr_time
            return True
        if self.in_local_navi_:
            if curr_time - self.time_local_navi < 1:
                return True
            return not self.local_navigation_.is_back_to_path(position[:2])
        return False

    def get_time(self):
        return time.time()

    def iterate(self):
        '''
        Implementation of a single iteration of the control loop.

        Returns:
            np.array: pose estimate of current iteration
        '''
        curr_time = self.get_time()

        curr_u = self.previous_u # speed of wheels in grid units
        curr_sensors = self.robot_.read_sensors()  # proximity sensors
        
        self.vision_.estimate_position()
        
        # outputs (x,y) pos of thymio in pixels, angle with horizontal in degrees
        position_measurement, angle_measurement = self.vision_.get_pose_thymio_grid()

        # Discard or keep measurement for filtering
        if position_measurement is None or angle_measurement is None:
            pose_measurement = None
        else:
            pose_measurement = [
                position_measurement[0],
                position_measurement[1],
                angle_measurement,
            ]

        pose_estimate = self.filter_.iterate(curr_u, curr_time, pose_measurement)

        # local navigation / Controller Box
        if self.local_navigation_is_needed(pose_estimate, curr_sensors, curr_time):
            self.controller_.update_local_goal((pose_estimate[0], pose_estimate[1]))

            speed = self.local_navigation_.iterate(curr_sensors, pose_estimate)
            if not speed:
                print("STOP CAUSE STUCK")
                return self.global_navigation_.goal
            self.in_local_navi_ = True

        else:
            speed = self.controller_.iterate(
                (pose_estimate[0], pose_estimate[1]), pose_estimate[2]
            )
            self.in_local_navi_ = False
            self.local_navigation_.chosen_direct = None

        # Push to plant
        self.robot_.set_speed(speed)
        
        
        self.pt.print()

        # update variables
        self.previous_u = np.array(speed) / 25 / self.vision_.get_scaling_factor() # grid units

        return pose_estimate

    def follow_path(self):
        '''
        Implementation of the while loop of the Motion Control system.
        '''

        self.estimates = []
        self.measurements = []
        self.vision_.estimate_position()

        # initialize filter
        (
            initial_position_estimate,
            initial_angle_estimate,
        ) = self.vision_.get_pose_thymio_grid()
        initial_pose_estimate = [
            initial_position_estimate[0],
            initial_position_estimate[1],
            initial_angle_estimate,
        ]
        self.filter_.initialize(self.get_time(), initial_pose_estimate)

        # start loop
        position = self.iterate()
        while (
            not self.global_navigation_.at_goal(position[0:2])
            and self.vision_.is_camera_open()
        ):
            position = self.iterate()

        return
