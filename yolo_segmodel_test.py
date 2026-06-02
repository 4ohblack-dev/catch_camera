from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path



def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir/"best_seg.pt"
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)

    if not cap.isOpened():
        print("cannot find webcam")
        exit()
    
    while True:
        ret,frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame,project=str(base_dir),name="latest_testlog",exist_ok=True,stream=True,verbose=False,conf=0.6)
        results_plot = next(r.plot() for r in results)
        cv2.imshow("webcam seg predicted",results_plot)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
