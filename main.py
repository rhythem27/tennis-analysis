from utils import (read_video, save_video, measure_distance, draw_player_stats, convert_meters_to_pixel_distance)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import pandas as pd
import cv2
from copy import deepcopy

def main():
    # Read video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Track players and balls
    player_tracker = PlayerTracker("models/yolov8n.pt")
    ball_tracker = BallTracker("models/yolov8n.pt")
    player_detection = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detection.pkl")
    ball_detection = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detection.pkl")

    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)

    # Detect court lines
    court_model_path = "training/tennis_court_keypoint.pt"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detection = player_tracker.choose_and_filter_players(court_keypoints, player_detection)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0])

    # detect ball shots 
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detection)
    print(ball_shot_frames)

    # convert positions to mini court
    player_mini_court_detection, ball_mini_court_detection = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detection, ball_detection, court_keypoints)

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    for ball_shot_ind in range (len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_sec = (end_frame - start_frame)/24    # 24 fps

        # get distance covered by ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detection[start_frame][1], ball_mini_court_detection[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())

        # speed of ball in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_sec * 3.6

        # player who shot the ball
        player_positions = player_mini_court_detection[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                               ball_mini_court_detection[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detection[start_frame][opponent_player_id],
                                                               player_mini_court_detection[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_meters_to_pixel_distance(distance_covered_by_opponent_pixels, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court())

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_sec * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent
        
        player_stats_data.append(current_player_stats) 

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.fillna(0) 

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots'] # Div by opponent's shots
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots'] # Div by opponent's shots
    player_stats_data_df = player_stats_data_df.fillna(0)

    # Draw output

    ## Draw bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detection)

    ## Draw court lines
    output_video_frames = court_line_detector.keypoints_on_video(output_video_frames, court_keypoints)

    ## Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames, player_mini_court_detection, ball_mini_court_detection)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detection, color=(0, 255, 0))
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detection, color=(0, 255, 255))

    # draw player stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 225), 2)
    

    save_video(output_video_frames, "output_videos/output_video.avi")
    

if __name__ == "__main__":
    main()