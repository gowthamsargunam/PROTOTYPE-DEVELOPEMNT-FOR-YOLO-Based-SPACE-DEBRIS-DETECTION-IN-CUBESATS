# PROTOTYPE-DEVELOPEMNT-FOR-YOLO-Based-SPACE-DEBRIS-DETECTION-IN-CUBESATS
Space Debris Detection Using Yolov5 Nano and A Streamlit Alert System for Debris Detection

# Installation - YOLOv5 and Torch (GPU Version)

```
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!git clone https://github.com/ultralytics/yolov5
!cd yolov5 && pip install -r requirements.txt

```
# Check CUDA Availability
```python
import torch
print(torch.cuda.is_available())

```
# Model Loading and Object Detection (to check the model working properly and object is detecting or not)
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5n') #a suggested model yolov5n(nano)
model.conf = 0.1
results = model('image.jpg')
results.print()

# visualize the object detected images
%matplotlib inline 
plt.imshow(np.squeeze(results.render()))
plt.show()

results.show()

```

### Make Your Custom or pre-trained/pre-collected image dataset of Space Debris to implement in the yolov5n model

Using custom dataset? follow the below steps:

1.Rename the images using the format: <label>_<UUID>.<image_format>, where <label> corresponds to the appropriate class based on the image content, <UUID> is a unique identifier, and <image_format> is the image extension (e.g., .jpg, .png, etc...)

2.Install labelImg 
```bash
!git clone https://github.com/HumanSignal/labelImg
!pip install pyqt5 lxml --upgrade
!cd labelImg && pyrcc5 -o libs/resources.py resources.qrc
```
3.Open the labelImg tool to create bounding boxes
```bash
cd labelImg
python labelImg.py # using command or git prompt
```

# Training the yolo model
```bash
!cd yolov5 && python train.py --img 640 --batch 8 --epochs 100 --data dataset.yaml --weights yolov5n.pt --workers 2 --cache ram/disk

```
# Load The Model
```python
import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\Gowth\yolov5\runs\train\exp\weights\best.pt')

#Lower confidence threshold (default=0.25)
model.conf = 0.1

model.iou = 0.45  # Merge overlapping detections more strictly

```
Visualizing Space Debris through images

```python
results = model(r"you_space_debris_image.jpg")
results.print()
results.show()
```

Visualizing Space Debris through Video

```python
import cv2
import numpy as np

cap=cv2.VideoCapture(r"C:\Users\Gowth\data\debris.mp4")
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:break
    results=model(frame)
    cv2.imshow('YOLO',np.squeeze(results.render()))
    if cv2.waitKey(10)&0xFF==ord('q'):break
cap.release()
cv2.destroyAllWindows()

```
# Making a Streamlit Dashboard
A python file was already created, named as "stream.py" and saved. To open the Dashboard for the Detection purpose to show detection events and detection logs as a Real-time Alerts

to run that:

Run the Python file as:
```bash
!streamlit run stem.py













