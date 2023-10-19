import pyautogui
import numpy as np 
import cv2
def get_screenshot():
    screencap = pyautogui.screenshot()
    screencap = np.array(screencap)#Load to queue ~
    # screencap = screencap.resize(1, 2)
    screencap = cv2.resize(screencap, (28, 28))
    print(screencap.shape)