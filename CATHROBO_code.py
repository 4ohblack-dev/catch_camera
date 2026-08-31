import cv2 
from ultralytics import YOLO
import numpy as np
import serial
import struct
import pygame

SCREEN_WIDTH=1280.0
SCREEN_HEIGHT = 720.0
CENTER_X = 640.0
CENTER_Y = 360.0

SERIAL_PORT = "COM5"
BAUDRATE = 115200

HEADER = b'\xAA'
DATA_FORMAT = '<2f6b3fb'
DATA_SIZE = struct.calcsize(DATA_FORMAT)
PACKET_SIZE = len(HEADER) + DATA_SIZE + 1

def deadzone(v,threshold=0.05):
    return 0 if abs(v) < threshold else v


def calculateCRC(data: bytes) -> int:
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
    cap= cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,SCREEN_HEIGHT)
    if not cap.isOpened():
        print("Cannot open the camera")
        return

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("Controller not found")
        cap.release()
        pygame.quit()
        return
    try:
        ser = serial.Serial(SERIAL_PORT,BAUDRATE,timeout=0.05)
    except serial.SerialException as e:
        ser = None        

    joy = pygame.joystick.Joystick(0)
    joy.init()
    
    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("Cannot get img")
            break

        pygame.event.pump()
        deltaX=deltaY=angle=0.0
        state = 0
        largest_contour = None
        results = model.predict(frame,stream=False,verbose=False,conf=0.6,retina_masks=True)
        r=results[0]
        masks=r.masks
        if r.masks is not None:
            masks_data = masks.data.cpu().numpy()
            for mask in masks_data:
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
                    state = 1 if (deltaX*deltaX + deltaY*deltaY)< 100*100 else 0
                    print("〇") if state == 1 else print("✕")
        leftY = -deadzone(joy.get_axis(1))
        rightY = -deadzone(joy.get_axis(3))
        L_button = 1 if joy.get_button(9) else 0
        R_button = 1 if joy.get_button(10) else 0
        button_left  = 1 if joy.get_button(13) == 1 else 0
        button_right = 1 if joy.get_button(14) == 1 else 0
        button_up    = 1 if joy.get_button(11) == 1 else 0
        button_down  = 1 if joy.get_button(12) == 1 else 0
        values = [leftY,rightY,
                  L_button,R_button,
                  button_left,button_right,button_up,button_down,
                  deltaX,deltaY,angle,state
                  ]
        data_bytes = struct.pack(DATA_FORMAT,*values)
        crc = calculateCRC(data_bytes)
        packet = HEADER + data_bytes + bytes([crc])
        if ser is not None:
            ser.write(packet)
            head = ser.read(1)

            if len(head) == 0:
                print("No signal from ESP32")

            elif head != HEADER:
                print(f"Unknown header received: {head.hex()}")

                if ser.in_waiting > 0:
                    ser.reset_input_buffer()

            else:
                read_bytes = ser.read(DATA_SIZE + 1)

                if len(read_bytes) != DATA_SIZE + 1:
                    print(
                        f"Invalid data length: "
                        f"{len(read_bytes)}/{DATA_SIZE + 1}"
                    )
                else:
                    read_data_bytes = read_bytes[:DATA_SIZE]
                    read_crc = read_bytes[DATA_SIZE]

                    calc = calculateCRC(read_data_bytes)

                    if calc != read_crc:
                        print(
                            f"CRC Error: "
                            f"received={read_crc:02X}, "
                            f"calculated={calc:02X}"
                        )
                    else:
                        unpacked = struct.unpack(DATA_FORMAT,read_data_bytes)

                        print(
                            f"Success! -> "
                            f"leftY:{unpacked[0]:.2f}, "
                            f"rightY:{unpacked[1]:.2f}, "
                            f"L:{unpacked[2]}, "
                            f"R:{unpacked[3]}, "
                            f"LEFT:{unpacked[4]}, "
                            f"RIGHT:{unpacked[5]}, "
                            f"UP:{unpacked[6]}, "
                            f"DOWN:{unpacked[7]}, "
                            f"X:{unpacked[8]:.2f}, "
                            f"Y:{unpacked[9]:.2f}, "
                            f"RO:{unpacked[10]:.2f}, "
                            f"STATE:{unpacked[11]}"
                        )


if __name__ == "__main__":
    main()