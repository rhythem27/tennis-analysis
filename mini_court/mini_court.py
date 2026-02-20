import cv2
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import constants
from utils import (convert_pixel_distance_to_meters, convert_meters_to_pixel_distance, get_foot_position, get_closest_keypoint_index, get_height_of_bbox, measure_xy_distance, get_center_of_bbox, measure_distance)

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
        return convert_meters_to_pixel_distance(meters, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)

    def convert_pixels_to_meters(self, pixels):
        return convert_pixel_distance_to_meters(pixels, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

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

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # draw lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

        # draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2)) 
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (0, 0, 255), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        #draw rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        masks = shapes.astype(bool)
        out[masks] = cv2.addWeighted(frame[masks], alpha, shapes[masks], 1 - alpha, 0).squeeze()
        return out

    def draw_mini_court(self, frames, player_detections, ball_detections):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)

            # Draw players
            for player_id, position in player_detections[frame_num].items():
                x, y = int(position[0]), int(position[1])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Draw ball
            if 1 in ball_detections[frame_num]:
                position = ball_detections[frame_num][1]
                x, y = int(position[0]), int(position[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    def width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self, object_position, closest_keypoint, closest_keypoint_index, player_height_in_pixels, player_height_in_meters):
        distance_from_keypoint_x_pixels = object_position[0] - closest_keypoint[0]
        distance_from_keypoint_y_pixels = object_position[1] - closest_keypoint[1]

        # convert pixel distance to meters 
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels, player_height_in_meters, player_height_in_pixels)
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels, player_height_in_meters, player_height_in_pixels)

        # convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_pixels(distance_from_keypoint_y_meters)
        
        closest_mini_court_keypoint = (self.drawing_key_points[closest_keypoint_index*2], self.drawing_key_points[closest_keypoint_index*2+1])
        
        mini_court_player_position = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels, 
                                      closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)
        return mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_keypoints):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox_dict in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num].get(1)
            
            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox_dict.items():
                foot_position = get_foot_position(bbox)

                # Use a specific set of keypoints for conversion (e.g. baseline corners)
                closest_keypoint_index = get_closest_keypoint_index(foot_position, original_court_keypoints, [0, 2, 12, 13])
                closest_keypoint = (original_court_keypoints[closest_keypoint_index*2], original_court_keypoints[closest_keypoint_index*2+1])

                # Get the max height of the player in nearby frames to get a better estimate
                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)
                
                heights = []
                for i in range(frame_index_min, frame_index_max):
                    if player_id in player_boxes[i]:
                        heights.append(get_height_of_bbox(player_boxes[i][player_id]))
                
                max_player_height_in_pixels = max(heights) if heights else get_height_of_bbox(bbox)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position, closest_keypoint, closest_keypoint_index, max_player_height_in_pixels, player_heights.get(player_id, 1.85))
                output_player_bboxes_dict[player_id] = mini_court_player_position
                
                if ball_box is not None:
                    ball_position = get_center_of_bbox(ball_box)
                    closest_player_id_to_ball = min(player_bbox_dict.keys(), key=lambda x: measure_distance(ball_position, get_foot_position(player_bbox_dict[x])))
                    
                    if closest_player_id_to_ball == player_id:
                        mini_court_ball_position = self.get_mini_court_coordinates(ball_position, closest_keypoint, closest_keypoint_index, max_player_height_in_pixels, player_heights.get(player_id, 1.85))
                        output_ball_boxes.append({1: mini_court_ball_position})

            output_player_boxes.append(output_player_bboxes_dict)
            
            if len(output_ball_boxes) <= frame_num:
                output_ball_boxes.append({})

        return output_player_boxes, output_ball_boxes  


    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1) 

        return frames