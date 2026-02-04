from ultralytics import YOLO

# Load the YOLO 

model = YOLO("yolov10n.pt")
# model = YOLO("C:/Tennis Analysis/training/runs/detect/train2/weights/last.pt")

results = model.track('input_videos/input_video.mp4',conf=0.2, save=True)
# print(results)
# print("boxes:")
# for box in results[0].boxes:
#     print(box)



