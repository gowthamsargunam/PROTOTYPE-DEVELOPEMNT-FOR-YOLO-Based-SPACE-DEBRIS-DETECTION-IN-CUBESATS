# PROTOTYPE-DEVELOPEMNT-FOR-YOLO-Based-SPACE-DEBRIS-DETECTION-IN-CUBESATS
Space Debris Detection Using Yolov5 Nano and A Streamlit Alert System for Debris Detection

# Step1:
Install the need Dependencies

install pytorch
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # i have GPU so I'm using CUDA version "CPU Version also available"

Colne the YOLOV5 repository in our local machine
!git clone https://github.com/ultralytics/yolov5
!cd yolov5 && pip install -r requirements.txt

To verify if PyTorch can leverage GPU acceleration, if True CUDA is available, not available gives False
!python -c "import torch; print(torch.cuda.is_available())"


# Step2:
verify the yolov5 model working properly, To check if the model is correctly detecting objects

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

A Suggested Model YOLOV5n(nano)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.conf = 0.1
model

to visualize the image that the objects are detected
img = r"your_image_path"
results = model(img)
results.print()

%matplotlib inline 
plt.imshow(np.squeeze(results.render()))
plt.show()
