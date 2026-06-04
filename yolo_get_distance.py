import cv2 
from ultralytics import YOLO
import numpy as np



def main():
    F_length = 100.0
    real_width = 10.0
    model = YOLO("best_seg.pt")
    cap= cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    while cap.Opened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = model.predict(frame,stream=True,varbose=False,conf=0.6)
            for r in results:
                masks=r.masks
                boxes=r.boxes

                if masks is not None:
                    for box,mask in zip(boxes,masks):
                        class_id= int(box.cls[0])

    
        
         
             
             


    




if __name__ == "__main__":
    main()