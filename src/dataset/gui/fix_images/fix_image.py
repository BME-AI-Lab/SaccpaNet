import cv2
from .generate_image import RGBImageHandler, DepthImageHandler
import numpy as np
import tqdm
from .get_image_list import get_image_list

current_image_index = 0
current_joint = 0


def click_and_crop(event, x, y, flags, param):
    x = x / 3
    y = y / 3
    # print(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        set_patch(x, y)
        update_image()
    elif event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        set_patch(x, y)
        update_image()
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        print(x, y)
        set_patch(x, y)
        update_image()
    elif event == cv2.EVENT_MBUTTONUP:
        # imageResover.set_patch(current_joint,x,y)
        # update_image()
        capture(1)
    elif event == cv2.EVENT_MOUSEWHEEL:
        # print(param)
        global current_joint
        if flags > 0:
            current_joint = (current_joint + 1) % 19
            set_joint(current_joint)
        else:
            current_joint = (current_joint - 1) % 19
            set_joint(current_joint)


def set_patch(x, y):
    imageResover.set_patch(current_joint, x, y)
    # imageResoverRGB.set_patch(current_joint, x, y)


def update_joint(x):
    global current_joint
    current_joint = cv2.getTrackbarPos("Joint", "image")


def set_joint(x):
    cv2.setTrackbarPos("Joint", "image", current_joint)
    print(current_joint)


next = 0


def capture(x):
    global next
    next = 1  # cv2.getTrackbarPos("capture","image")
    print("captured")
    # if next:
    imageResover.commit_patch(current_image_index)
    # imageResover


def update_image():
    global image, imageRGB
    image = imageResover.get(current_image_index)
    image = cv2.resize(image, np.array(image.shape[::-1])[1:] * 3)
    cv2.imshow("image", image)
    # imageRGB = imageResoverRGB.get(current_image_index)
    # imageRGB = cv2.resize(imageRGB, np.array(imageRGB.shape[::-1])[1:] * 3)
    # cv2.imshow("RGB", imageRGB)
    # cv2.waitKey(1000)


image_database = {
    "connection": "mysql+pymysql://root:bmepolyu@nas.polyu.eu.org/bmepolyu",
    "database": "data_06_15_images",
}
image = np.zeros((1, 1))
if __name__ == "__main__":
    cv2.namedWindow("image")
    # cv2.namedWindow("RGB")
    cv2.setMouseCallback("image", click_and_crop)
    # cv2.setMouseCallback("RGB", click_and_crop)
    cv2.createTrackbar("Joint", "image", 0, 18, update_joint)
    cv2.setTrackbarPos("Joint", "image", 18)
    cv2.createTrackbar("capture", "image", 0, 1, capture)
    imageResover = RGBImageHandler()
    imageResover.initialize(**image_database)
    # imageResoverRGB = RGBImageHandler()
    # imageResoverRGB.initialize(**image_database)

    # manual edit
    customList = [2408]
    imageList = list(zip(customList, [16] * len(customList)))

    # list edit
    # imageList = list(get_image_list())
    # starting_from = 110+130+29+97+24+16+7+74+102 +72+136+133+43+132+52+127+133+96+128
    # imageList = imageList[starting_from:]
    # imageList = imageList[58:]
    current = 0
    for _ in tqdm.tqdm(imageList):
        i, j = imageList[current]
        current_image_index = i  # int(input("Please enter the image id:"))
        current_joint = j
        cv2.setTrackbarPos("joint", "image", current_joint)
        # print(i, j, current)
        update_image()
        set_joint(current_joint)
        while not next:
            k = cv2.waitKey(1000)
            # print("wait...")
            if k == ord("a"):
                next = 1
                capture(1)
                break
            elif k == ord("s"):
                current -= 2
                break
            elif k == -1:
                pass
            else:
                print(k)
            # update_image()
        next = 0
        current += 1
