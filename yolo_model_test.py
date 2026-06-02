import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from pathlib import Path
import numpy as np


base_dir = Path(__file__).resolve().parent
model_path = base_dir/"best.pt"

model = YOLO(str(model_path))
test_imgs_path = base_dir/"test_img"

if not os.path.exists(test_imgs_path):
    print("error")

results = model.predict(test_imgs_path,save=True,project="test",name="")
print(len(results))
test_result = results[3]
testimg = cv2.cvtColor(test_result.plot(),cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.imshow(testimg)
plt.axis("off")
plt.show()
