from ultralytics import YOLO

# Model
#model = torch.hub.load('ultralytics/ultralytics', 'yolov8n', pretrained=True)
model = YOLO("yolov8x.pt")

# Images
imgs = [r"C:\Datasets\faiss - Copy\pics 2\Polar-bears-ice-floe-Norway.webp",
        r"C:\Datasets\faiss - Copy\pics 2\merlion.jpg",
        r"C:\Datasets\faiss - Copy\pics 2\dog (3).JPEG",
        r"C:\Datasets\faiss\pics 2\dog (10).JPEG",
        r"C:\Datasets\faiss\pics 2\fish (5).JPEG"]  # batch of images

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


# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
