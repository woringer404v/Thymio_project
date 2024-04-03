import cv2
import time
import numpy as np
import math


B_UP = np.array([110, 255, 255])
B_LOW = np.array([90, 40, 40])
G_UP = np.array([80, 255, 255])
G_LOW = np.array([60, 40, 40])
R_UP = np.array([200, 255, 255])
R_LOW = np.array([170, 40, 40])
MAP_REAL_WIDTH = 74
MAP_REAL_LENGTH = 109


class Vision:
    '''
    Module that implements the Vision part of the project.
    Out of these functions, to use them in the loop, the estimate_position() needs
    to be called to update the values, and then the others are called to obtain them.

    If there is a problem with the camera and we are sure that the values that we got
    are not accurate, we return a None. This can then be seen by the filter which then 
    ignores this data.
    '''
    def __init__(self, img=None) -> None:
        self.cap = cv2.VideoCapture(0)
        self.hsv_img = None
        self.pos_green = None
        self.pos_blue = None
        self.goal = None
        self.pos_thymio = None
        self.angle_thymio = None
        self.rescaled_pos = None
        self.rescaled_goal = None
        self.planar_img = None
        self.positions = []
        self.green_positions = []
        self.scale = 0.05

    def is_camera_open(self):
        return self.cap.isOpened()

    def close_camera(self):
        self.cap.release()

    def get_pose_thymio_pixels(self):
        return self.pos_thymio, self.angle_thymio

    def get_pose_thymio_grid(self):
        return self.rescaled_pos, self.angle_thymio

    def get_scale_grid_to_cm(self):
        scale_width = round((MAP_REAL_WIDTH / self.maze.shape[0]), 1)
        scale_length = round((MAP_REAL_LENGTH / self.maze.shape[1]), 1)
        if scale_width != scale_length:
            print("Can't scale from grid to cm")
            print(scale_width, scale_length)

        else:
            return scale_length

    def take_init_picture(self):
        """
        Takes the first picture for initialization, waits before taking it to let camera focus

        """
        for i in range(2):
            ret, picture = self.cap.read()
            if ret == False:
                print("Camera not reading")
            time.sleep(2)  # wait 2 seconds --> necessary ????
        self.raw_img = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        self.hsv_img = cv2.cvtColor(self.raw_img, cv2.COLOR_RGB2HSV)

    def take_picture(self):
        ret, picture = self.cap.read()
        if ret == False:
            print("Camera is not reading")
        else:
            self.raw_img = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
            self.hsv_img = cv2.cvtColor(self.raw_img, cv2.COLOR_RGB2HSV)

    def initialize_maze(self):
        self.take_init_picture()
        self.find_corners()
        self.img_to_planar()
        self.goal = (
            round(self.detect_circle(R_LOW, R_UP)[0]),
            round(self.detect_circle(R_LOW, R_UP)[1]),
        )
        self.update_pose_thymio()
        self.find_obstacles()
        self.rescale_positions()
        (x, y) = self.rescaled_pos  # changing y and x
        (x_goal, y_goal) = self.rescaled_goal

        return (
            self.obs_to_maze(),
            (round(x), round(y)),
            self.angle_thymio,
            (round(x_goal), round(y_goal)),
        )

    def estimate_position(self):  # Call it for every main loop
        self.take_picture()
        try:
            self.find_corners()
        except IndexError:  # Did not find corners
            self.pos_thymio = None
            self.rescaled_pos = None
            self.angle_thymio = None
            self.pos_green = None
            self.pos_blue = None
            return
        self.img_to_planar()
        self.update_pose_thymio()
        self.rescale_positions()

    def detect_circle(self, color_low, color_up):
        self.hsv_img = cv2.cvtColor(self.planar_img, cv2.COLOR_RGB2HSV)

        # reduce noise
        mask = cv2.inRange(self.hsv_img, color_low, color_up)
        mask = cv2.erode(mask, None, iterations=6)
        mask = cv2.dilate(mask, None, iterations=6)
        # detect contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # find the position and radius of the circle
        contours_max = max(contours, key=cv2.contourArea)

        # if len(contours) != 1:
        #     print("Noise while detecting circle")
        # Still take the first contour, probably is the good one
        (pos, _) = cv2.minEnclosingCircle(contours_max)

        return pos

    def update_pose_thymio(self):
        """
        Get Thymio's position (x,y) and angle w.r.t the horizontal axis

        Input: RGB (planar) image of pixels
        Output: position (x,y) in pixels, angle in degrees
        """
        try:
            # BLUE CIRCLE
            self.pos_blue = self.detect_circle(B_LOW, B_UP)

            # GREEN CIRCLE
            self.pos_green = self.detect_circle(G_LOW, G_UP)
            self.green_positions.append(
                (round(self.pos_green[0]), round(self.pos_green[1]))
            )
        except ValueError:
            self.pos_thymio = None
            self.angle_thymio = None
            print("DID NOT DETECT CIRCLE BLUE/GREEN")
            return
        # TODO: put it in print

        # FIND ROBOT POSE FROM BLUE AND GREEN CIRCLES
        self.pos_thymio = self.pos_blue
        delta_x = self.pos_green[0] - self.pos_blue[0]
        delta_y = self.pos_green[1] - self.pos_blue[1]
        self.angle_thymio = (np.arctan2(delta_y, delta_x)) * (
            180 / np.pi
        )  # minus for delta_y because y increases from top to bottom

        self.positions.append((round(self.pos_thymio[0]), round(self.pos_thymio[1])))
        if math.dist(self.pos_blue, self.pos_green) > 5 / self.scale:
            print("CIRCLES ARE TOO SEPARATE")
            self.pos_thymio = None
            self.angle_thymio = None

    # TODO: maybe it is better to have round instead of int?
    def rescale_positions(self):
        if self.pos_thymio is not None:
            self.rescaled_pos = (
                self.pos_thymio[0] * self.scale,
                self.pos_thymio[1] * self.scale,
            )
        else:
            self.rescaled_pos = None
        self.rescaled_goal = (self.goal[0] * self.scale, self.goal[1] * self.scale)

    def find_corners(self):
        """
        Finds the 4 corners of the map contour

        Input: RGB image of pixels
        Output: corners in the order: bottom_left, bottom_right, top_left, top_right
        """
        # Threshold to get only the black shapes
        bilateral = cv2.bilateralFilter(self.raw_img, 9, 75, 75)
        bw_img = cv2.cvtColor(bilateral, cv2.COLOR_RGB2GRAY)
        ret, th1 = cv2.threshold(bw_img, 70, 255, cv2.THRESH_BINARY)
        th1 = cv2.erode(th1, None, iterations=4)
        th1 = cv2.dilate(th1, None, iterations=4)

        # Detect the contours
        ext_contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate the contours to have less noise
        approx_c = []
        for c in ext_contours:
            epsilon = 0.016 * cv2.arcLength(c, True)
            approx_c.append(cv2.approxPolyDP(c, epsilon, True))

        # Order the contours by area
        areas = []
        for c in approx_c:
            areas.append(cv2.contourArea(c))
        sorted_contours_ind = np.argsort(areas)
        # Map contour is the contour with the second biggest area
        map_contour_ind = sorted_contours_ind[-2]

        # Find the 4 corners of the map contour
        map_contour = approx_c[map_contour_ind]
        # Top left and bottom left corners have the smallest and biggest sum of coordinates respectively
        sum_map_corners = np.sum(map_contour, axis=2)
        top_left = map_contour[np.argmin(sum_map_corners)][0]
        bottom_right = map_contour[np.argmax(sum_map_corners)][0]
        # Top right and bottom right corners have smallest and biggest difference y-x respectively
        diff_map_corners = np.diff(map_contour, axis=2)
        top_right = map_contour[np.argmin(diff_map_corners)][0]
        bottom_left = map_contour[np.argmax(diff_map_corners)][0]

        self.corners = [bottom_left, bottom_right, top_left, top_right]

    def resize_img(self, image):
        """
        Size the image down to get smaller image dimensions

        Input: Image of pixels, scale = percentage of the previous image
        Output: Resized image
        """
        # Size the image down
        new_dim = (
            int(image.shape[1] * self.scale),
            int(image.shape[0] * self.scale),
        )
        img_down = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

        return img_down

    def img_to_planar(self):
        """
        Get a top view of the image from 4 reference corner points.

        Input: RGB image of pixels, array of corner coordinates
        Output: Planar/top view image of pixels
        """
        bottom_left, bottom_right, top_left, top_right = self.corners[
            :
        ]  # Check if works fine

        # Compute width of grid
        bottom_width = np.sqrt(
            (bottom_right[0] - bottom_left[0]) ** 2
            + (bottom_right[1] - bottom_left[1]) ** 2
        )
        top_width = np.sqrt(
            (top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2
        )
        width = int(np.max([bottom_width, top_width]))

        # Compute length of grid
        left_length = np.sqrt(
            (bottom_left[0] - top_left[0]) ** 2 + (bottom_left[1] - top_left[1]) ** 2
        )
        right_length = np.sqrt(
            (bottom_right[0] - top_right[0]) ** 2
            + (bottom_right[1] - top_right[1]) ** 2
        )
        length = int(np.max([left_length, right_length]))

        # Conversion to planar view
        source_corners = np.array(
            [bottom_left, bottom_right, top_left, top_right], dtype=np.float32
        )
        output_corners = np.array(
            [[0, length], [width, length], [0, 0], [width, 0]], dtype=np.float32
        )

        transformation = cv2.getPerspectiveTransform(source_corners, output_corners)
        self.planar_img = cv2.warpPerspective(
            self.raw_img, transformation, (width, length)
        )

    def find_obstacles(self):
        # Bilateral filtering
        bilateral = cv2.bilateralFilter(self.planar_img, 9, 75, 75)
        # Binary image
        bw_img = cv2.cvtColor(bilateral, cv2.COLOR_RGB2GRAY)
        # Threshold to keep only the obstacles
        ret, obstacles_img = cv2.threshold(bw_img, 90, 255, cv2.THRESH_BINARY)
        # Resize the image
        obstacles_img = self.resize_img(obstacles_img)
        # Remove noise  ## CHECK IF NECESSARY !! (FOR SMALL BLACK SENSORS ON THYMIO)
        obstacles_img = cv2.medianBlur(obstacles_img, ksize=3)

        self.obstacles_img = obstacles_img

    def obs_to_maze(self):
        # Transform into grid of 0 (black) and 1 (white)
        self.maze = np.array(self.obstacles_img)
        self.maze[self.maze != 0] = 1

        return np.transpose(1 - self.maze)

    def print(
        self,
        path,
        local_goal,
    ):
        to_print = self.planar_img

        # Print thymio's trajectory
        for i, p in enumerate(self.positions):
            cv2.circle(
                to_print,
                p,
                radius=int(1),
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.circle(
                to_print,
                self.green_positions[i],
                radius=int(1),
                color=(0, 255, 0),
                thickness=2,
            )

        cv2.circle(
            to_print,
            self.goal,
            radius=int(1),
            color=(0, 0, 255),
            thickness=2,
        )
        cv2.circle(
            to_print,
            (int(local_goal[0] / self.scale), int(local_goal[1] / self.scale)),
            radius=int(6),
            color=(0, 0, 255),
            thickness=2,
        )

        # Rescale and print the global path
        for i in range(path.shape[1] - 1):
            cv2.line(
                to_print,
                (int(path[0, i] / self.scale), int(path[1, i] / self.scale)),
                (int(path[0, i + 1] / self.scale), int(path[1, i + 1] / self.scale)),
                (0, 0, 0),
                2,
            )
            cv2.circle(
                to_print,
                (int(path[0, i] / self.scale), int(path[1, i] / self.scale)),
                radius=int(3),
                color=(0, 0, 0),
                thickness=2,
            )

        # Print the Kalman estimation #[TODO]

        cv2.imshow("frame", to_print)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return 0
