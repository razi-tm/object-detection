from ultralytics import YOLO

# Model
model = YOLO("yolov8x.pt")        #yolov5x    yolov5n6    yolov8n    ...

# Images
imgs = [r"C:\Datasets\pics\Polar-bears-ice-floe-Norway.webp",
        r"C:\Datasets\pics\merlion.jpg",
        r"C:\Datasets\pics\dog (3).JPEG",
        r"C:\Datasets\pics\dog (10).JPEG",
        r"C:\Datasets\pics\fish (5).JPEG"]  # batch of images

# Inference
#results = model.predict(imgs)
results = model(imgs)

#result = results[0]

for result in results:
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        print("Object type:", class_id)
        print("Coordinates:", cords)
        print("Probability:", conf)
    print("---")
