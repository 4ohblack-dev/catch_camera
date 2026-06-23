import cv2 
from ultralytics import YOLO
import numpy as np
import serial
import struct
import time
import datetime

SERIAL_PORT = "COM3"
BAUDRATE = 1152000

HEADER = b'\xAA'
DATA_FORMAT = '<fff'
DATA_SIZE = struct.calcsize(DATA_FORMAT)
PACKET_SIZE = len(HEADER) + DATA_SIZE + 1
F_LENGTH = 100.0
REAL_WIDTH = 10.0
SCREEN_WIDTH=1280.0
SCREEN_HEIGHT = 720.0
CENTER_X = 640.0
CENTER_Y = 360.0

def log_with_time(message):
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{current_time}] {message}")

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
    model = YOLO("best_seg_openvino_model",task="segment")
    cap= cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,SCREEN_HEIGHT)
    #ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
    time.sleep(2)
    newdistance = None
    alpha = 0.05
    State = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = model.predict(frame,stream=False,verbose=False,conf=0.6,retina_masks=True)
            r=results[0]
            masks=r.masks
            boxes=r.boxes

            if r.masks is not None and r.boxes is not None:
                masks_data = masks.data.cpu().numpy()
                for box,mask in zip(boxes,masks_data):
                    
                    mask_data = (mask*255).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(mask_data,cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour= max(contours,key=cv2.contourArea)
                        hull = cv2.convexHull(largest_contour)
                        rect = cv2.minAreaRect(hull)
                        (centerX,centerY) = rect[0]
                        angle = rect[2]
                        (w,h) = rect[1]
                        angle += 90 if w<h else 0
                        angle = angle % 180
                        deltaX,deltaY = centerX-CENTER_X,centerY-CENTER_Y
                        pixel_width = max(w,h)

                        State = True if (deltaX*deltaX + deltaY*deltaY)< 50*50 else False

                        print("〇") if State == True else print("✕")
                        
                        if pixel_width>0:
                            predistance = F_LENGTH*REAL_WIDTH/pixel_width

                            if newdistance is None:
                                newdistance=predistance
                            else:
                                newdistance=(1-alpha)*newdistance + alpha*predistance
#
#
#                            data_bytes = struct.pack(DATA_FORMAT,float(deltaX),float(deltaY),float(angle))
#                            crc = calculateCRC(data_bytes)
#                            packet = HEADER + data_bytes + bytes([crc])
#                            ser.write(packet)
#
#                            
                            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            text = f"Dist: {newdistance:.1f} cm"
                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)
#
#                            # --- 受信デバッグ用（タイムスタンプ付き） ---
#                            head = ser.read(1)
#
#                            if len(head) == 0:
#                                log_with_time(
#                                    "Waiting... (No data received from ESP32)"
#                                )
#                            elif head == HEADER:
#                                log_with_time("-> Header [0xAA] OK!")
#
#                                rx_bytes = ser.read(DATA_SIZE + 1)
#                                log_with_time(
#                                    f"-> Read remaining bytes: {len(rx_bytes)}/{DATA_SIZE+1} bytes"
#                                )
#
#                                if len(rx_bytes) == (DATA_SIZE + 1):
#                                    rx_data_bytes = rx_bytes[:DATA_SIZE]#先頭からDATASIZEだけ取り出す
#                                    rx_crc = rx_bytes[DATA_SIZE]#最後の1byteだけを取り出す
#
#                                    calc = calculateCRC(rx_data_bytes)
#                                    log_with_time(
#                                        f"-> CRC Check: Recv={rx_crc}, Calc={calc}"
#                                    )
#
#                                    if calc == rx_crc:
#                                        unpacked = struct.unpack(
#                                            DATA_FORMAT, rx_data_bytes
#                                        )
#                                        log_with_time(
#                                            f"✨ Success! -> dx: {unpacked[0]:.2f}, dy: {unpacked[1]:.2f}"
#                                        )
#                                    else:
#                                        log_with_time("❌ CRC Mismatch Error!")
#                                else:
#                                    log_with_time("❌ Packet fraction lost!")
#                            else:
#                                log_with_time(
#                                    f"-> Unknown byte received: {head.hex()}"
#                                )
#                                if ser.in_waiting > 0:
#                                    ser.reset_input_buffer()
#            
            cv2.imshow("Focal Length Calibrator (1280x720)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
    cv2.destroyAllWindows()
             
if __name__ == "__main__":
    main()