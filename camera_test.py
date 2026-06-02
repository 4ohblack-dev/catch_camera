import cv2

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)



cap.set(cv2.CAP_PROP_FPS,30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret,frame = cap.read()
    

    if not ret:
        print("Error: Can't receive frame.")
        break

    cv2.imshow('Webcam Feed',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()