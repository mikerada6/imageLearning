import os
from pathlib import Path
import cv2


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

focus = 100  # min: 0, max: 255, increment:5
cam.set(28, focus)

setCode = "eld"

img_counter = 0

dir = str(Path(os.getcwd()).parents[0])+"/tmp/captured/"+setCode
Path(dir).mkdir(parents=True, exist_ok=True)
print(os.getcwd())

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(dir+"/"+img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
