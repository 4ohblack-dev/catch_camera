import cv2 
from ultralytics import YOLO
import numpy as np



def main():
    F_length = 100.0
    real_width = 10.0
    model = YOLO("best_seg.onnx")
    cap= cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = model.predict(frame,stream=True,verbose=False,conf=0.6)
            for r in results:
                masks=r.masks
                boxes=r.boxes

                if masks is not None:
                    for box,mask in zip(boxes,masks):
                            mask_data = ((mask.data[0].cpu().numpy())*255).astype(np.uint8)
                            mask_data = cv2.resize(mask_data,(frame.shape[1],frame.shape[0]))
                            

                            contours, _ = cv2.findContours(mask_data,cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                 largest_contour= max(contours,key=cv2.contourArea)
                                 (_,_),radius = cv2.minEnclosingCircle(largest_contour)
                                 pixel_parameter = 2*radius

                                 if pixel_parameter>0:
                                    distance = F_length*real_width/pixel_parameter
                                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    text = f"Dist: {distance:.1f} cm"
                                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 0, 255), 2)
                cv2.imshow("Focal Length Calibrator (1280x720)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
    cv2.destroyAllWindows()
             
if __name__ == "__main__":
    main()