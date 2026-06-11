import cv2 
from ultralytics import YOLO
import numpy as np
import serial
import struct
import time

SERIAL_PORT = "COM3"
BAUDRATE = 1152000

HEADER = b'\xAA'
DATA_FORMAT = '<ff'
DATA_SIZE = struct.calcsize(DATA_FORMAT)
PACKET_SIZE = len(HEADER) + DATA_SIZE + 1
F_LENGTH = 100.0
REAL_WIDTH = 10.0
SCREEN_WIDTH=1280.0
SCREEN_HEIGHT = 720.0
CENTER_X = 640.0
CENTER_Y = 360.0

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
    model = YOLO("best_seg.onnx",task="segment")
    cap= cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,SCREEN_HEIGHT)
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.0)
    time.sleep(2)
    newdistance = None
    alpha = 0.05

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
                        (centerX,centerY) = rect[0]
                        angle = rect[2]
                        (w,h) = rect[1]
                        #deltaX,deltaY = centerX-CENTER_X,centerY-CENTER_Y
                        deltaX,deltaY = 10.0,10.0
                        pixel_width = max(w,h)

                        if pixel_width>0:
                            predistance = F_LENGTH*REAL_WIDTH/pixel_width

                            if newdistance is None:
                                newdistance=predistance
                            else:
                                newdistance=(1-alpha)*newdistance + alpha*predistance


                            data_bytes = struct.pack(DATA_FORMAT,float(deltaX),float(deltaY))
                            crc = calculateCRC(data_bytes)
                            packet = HEADER + data_bytes + bytes([crc])
                            ser.write(packet)
                            ser.flush()

                            
                            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            text = f"Dist: {newdistance:.1f} cm"
                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)

                            # --- 受信デバッグ用書き換え ---
                            head = ser.read(1)

                            if len(head) == 0:
                                # タイムアウトして1バイトも届かなかった場合
                                print("Waiting... (No data received from ESP32)")
                            elif head == HEADER:
                                print("-> Header [0xAA] OK!") # ヘッダーを見つけたログ

                                rx_bytes = ser.read(DATA_SIZE + 1)
                                print(f"-> Read remaining bytes: {len(rx_bytes)}/9 bytes") # バイト数ログ

                                if len(rx_bytes) == (DATA_SIZE + 1):
                                    rx_data_bytes = rx_bytes[:DATA_SIZE]
                                    rx_crc = rx_bytes[DATA_SIZE]

                                    calc = calculateCRC(rx_data_bytes)
                                    print(f"-> CRC Check: Recv={rx_crc}, Calc={calc}") # CRCの一致ログ

                                    if calc == rx_crc:
                                        unpacked = struct.unpack(DATA_FORMAT, rx_data_bytes)
                                        print(f"✨ Success! -> dx: {unpacked[0]:.2f}, dy: {unpacked[1]:.2f}")
                                    else:
                                        print("❌ CRC Mismatch Error!")
                                else:
                                    print("❌ Packet fraction lost!")
                            else:
                                # ヘッダーではない別のゴミデータが届いた場合
                                print(f"-> Unknown byte received: {head.hex()}")
                                if ser.in_waiting > 0:
                                    ser.reset_input_buffer()
            
            cv2.imshow("Focal Length Calibrator (1280x720)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
    cv2.destroyAllWindows()
             
if __name__ == "__main__":
    main()