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
# Model Loading and Object Detection
```
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.conf = 0.1
results = model('image.jpg')
results.print()


