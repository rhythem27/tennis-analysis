from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import pandas as pd
import cv2

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

    # Draw output

    ## Draw bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detection)

    ## Draw court lines
    output_video_frames = court_line_detector.keypoints_on_video(output_video_frames, court_keypoints)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 225), 2)
    

    save_video(output_video_frames, "output_videos/output_video.avi")
    

if __name__ == "__main__":
    main()