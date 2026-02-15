import cv2
import sys
import numpy as np
sys.path.append('../')
import constants
from utils import (convert_pixel_distance_to_meters, convert_meters_to_pixel_distance)

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectange_width = 250
        self.drawing_rectange_height = 450
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_pixels(self, meters):
        return convert_meters_to_pixel_distance(constants.HALF_COURT_LINE_HEIGHT*2, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

    def set_court_lines(self):
        self.lines = [
            (0,2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] , drawing_key_points[5] = int(self.court_start_x), int(self.court_end_y)
        # point 3
        drawing_key_points[6] , drawing_key_points[7] = int(self.court_end_x), int(self.court_end_y)
        # point 4
        drawing_key_points[8] , drawing_key_points[9] = int(self.court_start_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT))
        # point 5
        drawing_key_points[10] , drawing_key_points[11] = int(self.court_end_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT))
        # point 6
        drawing_key_points[12] , drawing_key_points[13] = int(self.court_start_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*2))
        # point 7
        drawing_key_points[14] , drawing_key_points[15] = int(self.court_end_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*2))
        # point 8
        drawing_key_points[16] , drawing_key_points[17] = int(self.court_start_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*3))
        # point 9
        drawing_key_points[18] , drawing_key_points[19] = int(self.court_end_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*3))
        # point 10
        drawing_key_points[20] , drawing_key_points[21] = int(self.court_start_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*4))
        # point 11
        drawing_key_points[22] , drawing_key_points[23] = int(self.court_end_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*4))
        # point 12
        drawing_key_points[24] , drawing_key_points[25] = int(self.court_start_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*5))
        # point 13
        drawing_key_points[26] , drawing_key_points[27] = int(self.court_end_x), int(self.court_start_y + self.convert_meters_pixels(constants.HALF_COURT_LINE_HEIGHT*5))

        self.drawing_key_points = drawing_key_points
        

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
        
    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.drawing_rectange_height + self.buffer
        self.start_x = self.end_x - self.drawing_rectange_width
        self.start_y = self.end_y - self.drawing_rectange_height
        
    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        #draw rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        masks = shapes.astype(bool)
        out[masks] = cv2.addWeighted(frame[masks], alpha, shapes[masks], 1 - alpha, 0)[masks]
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out
        