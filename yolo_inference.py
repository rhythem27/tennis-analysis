from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8x.pt")

results = model.predict('input_videos/input_video.mp4', save=True)
print(results)
print("boxes:")
for box in results[0].boxes:
    print(box)



