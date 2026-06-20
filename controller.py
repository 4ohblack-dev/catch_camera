import pygame
import time
import serial
import struct

SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 1152000

HEADER = b'\xAA'
DATA_FORMAT = '<fff'
DATA_SIZE = struct.calcsize(DATA_FORMAT)
PACKET_SIZE = len(HEADER) + DATA_SIZE + 1

def deadzone(v,threshold=0.05):
    return 0 if abs(v) < threshold else v

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
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()

    ser = serial.Serial(SERIAL_PORT,BAUDRATE,timeout=0.05)


    if count == 0:
        print("Controller not found")
        exit()

    joy = pygame.joystick.Joystick(0)
    joy.init()

    print("Name:",joy.get_name())
    print("Buttons:",joy.get_numbuttons())
    print("Axes:",joy.get_numaxes())

    while True:
        for event in pygame.event.get():
            print(event)

        leftX = deadzone(joy.get_axis(0))
        leftY = deadzone(joy.get_axis(1))
        rotL = deadzone(joy.get_axis(2))

        rightX = deadzone(joy.get_axis(3))
        rightY = deadzone(joy.get_axis(4))
        rotR = deadzone(joy.get_axis(5))

        #print(leftX)
        #print(leftY)
        #print(rotL)

        data_bytes = struct.pack(DATA_FORMAT,float(leftX),float(leftY),float(rotL))
        crc = calculateCRC(data_bytes)
        packet = HEADER + data_bytes + bytes([crc])
        ser.write(packet)


        head = ser.read(1)
        if len(head)==0:
            print("no signal from ESP32")
        elif head == HEADER:
            read_bytes = ser.read(DATA_SIZE+1)
            if len(read_bytes)== (DATA_SIZE+1):
                read_data_bytes = read_bytes[:DATA_SIZE]
                read_crc = read_bytes[DATA_SIZE]
                calc = calculateCRC(read_data_bytes)

                if calc == read_crc:
                    unpacked = struct.unpack(DATA_FORMAT,read_data_bytes)
                    print(f"Success! -> X:{unpacked[0]:.2f} , Y:{unpacked[1]:.2f} , RO:{unpacked[2]:.2f}")
        else:
            print("unknown byte recieved") 
            if ser.in_waiting >0:
                ser.reset_input_buffer()


#左     横：axes 0、縦：axes 1、ボタン：axes 2、L：button 4
#右     横：axes 3、縦：axes 4、ボタン：axes 5、R：button 5
#右十字 下から反時計回りに　button 0~3
#左十字 返り値が(,)で、正の方向が１、負の方向が‐1
#share : button 8
# options : button 9

if __name__ == "__main__":
    main()