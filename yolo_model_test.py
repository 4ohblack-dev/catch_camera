import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import time

def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir/"best.pt"
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)

    prev_time = 0

    if not cap.isOpened():
        print("cannot find cam")
        exit()

    while True:

        ret,frame = cap.read()
        if not ret:
            break
        
        
        results = model.predict(frame,project=str(base_dir),name="latest_testlog",exist_ok=True,stream=True,verbose=False,conf=0.6)
        results_plot = frame
        for r in results:
            results_plot = r.plot()

        cv2.imshow("webcam predicted", results_plot)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()