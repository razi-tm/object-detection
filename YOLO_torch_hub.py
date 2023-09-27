import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                                        #yolov5x  yolov5x6  ... 
# Images
imgs = [r"C:\Datasets\pics\Polar-bears-ice-floe-Norway.webp",
        r"C:\Datasets\pics\merlion.jpg",
        r"C:\Datasets\pics\dog (3).JPEG",
        r"C:\Datasets\pics\dog (10).JPEG",
        r"C:\Datasets\pics\fish (5).JPEG"]  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
#results.save()  # or .show()
results.show()

print(results.xyxy[0])  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
