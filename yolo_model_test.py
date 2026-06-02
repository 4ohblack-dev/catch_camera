import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from pathlib import Path
import numpy as np


base_dir = Path(__file__).resolve().parent
model_path = base_dir/"best.pt"
model = YOLO(str(model_path))

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS,30)

if not cap.isOpened():
    print("cannot find cam")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    capture_dir = Path(f"{base_dir}/captured.jpg")  
    if key ==ord(' '):
        
        if capture_dir.exists():
            capture_dir.unlink()

        cv2.imwrite(str(capture_dir),frame)

        break

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows

test_imgs_path = capture_dir

if not os.path.exists(test_imgs_path):
    print("predict error")
    exit()

results = model.predict(test_imgs_path,save=True,project=str(base_dir),name="latest_testlog",exist_ok=True)
#print(len(results))
test_result = results[0]
testimg = cv2.cvtColor(test_result.plot(),cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.imshow(testimg)
plt.axis("off")
plt.show()