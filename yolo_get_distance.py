import cv2 
from ultralytics import YOLO
import numpy as np
import serial
import struct
import time

SERIAL_PORT = "COM3"
BAUDRATE = 115200

HEADER = b'\xAA'
DATA_FORMAT = '<ff'
DATA_SIZE = struct.calcsize(DATA_FORMAT)
PACKET_SIZE = len(HEADER) + DATA_SIZE + 1


def calculateCRC(data: bytes) -> int:  # 引数はbytes型、戻り値はintで、1バイトの整数を返す
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def main():
    F_length = 100.0
    real_width = 10.0
    model = YOLO("best_seg.onnx",task="segment")
    cap= cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = model.predict(frame,stream=False,verbose=False,conf=0.6,retina_masks=True)
            r=results[0]
            masks=r.masks
            boxes=r.boxes

            if hasattr(r, 'masks') and masks is not None and hasattr(r, 'boxes') and boxes is not None:
                masks_data = masks.data.cpu().numpy()
                for box,mask in zip(boxes,masks_data):
                    
                    mask_data = (mask*255).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(mask_data,cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour= max(contours,key=cv2.contourArea)
                        hull = cv2.convexHull(largest_contour)
                        rect = cv2.minAreaRect(hull)
                        (w,h)=rect[1]
                        pixel_width = max(w,h)
                        newdistance = None
                        alpha = 0.05

                        if pixel_width>0:
                            predistance = F_length*real_width/pixel_width

                            if newdistance is None:
                                newdistance=predistance
                            else:
                                newdistance=(1-alpha)*newdistance + alpha*predistance

                            
                            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            text = f"Dist: {newdistance:.1f} cm"
                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)
            
            cv2.imshow("Focal Length Calibrator (1280x720)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
    cv2.destroyAllWindows()
             
if __name__ == "__main__":
    main()