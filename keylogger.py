from pynput.keyboard import Key, Listener
import logging
import os 
import numpy as np 
import pyautogui
import time 
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# 
logging.basicConfig(filename=(os.getcwd() + "/keylog.txt"), level=logging.DEBUG, format=" %(asctime)s - %(message)s")
count=0

class RecordingPipeline:

    def __init__(self) -> None:
        self.count = 0

    def on_press(self, key):
        logging.info(str(key))
        pyautogui.screenshot(os.getcwd() +  '/footage/' + str(count) + 'screenshot.png')
        self.count+=1 #Update image count on press action 
    

    def log_keystrokes(self):
        with Listener(on_press=self.on_press) as listener:
            listener.join()

def on_press(key):
    logging.info(str(key))
    img = pyautogui.screenshot(os.getcwd() +  '/footage/' + str(count) + 'screenshot.png')
    # pyautogui.screenshot(os.getcwd() +  '/footage/' + str(count)  + '.png')
    # count+=1


def log_keystrokes(func):
    with Listener(on_press=func) as listener:
        listener.join()
time.sleep(1) #Give me time to get ready
# print('GO!')
# log_keystrokes(on_press)
pipeline = RecordingPipeline()
pipeline.log_keystrokes()


'''
This logging block in keylog.txt is associated with each image keystroke.
    2023-10-16 02:07:31,481 - STREAM b'IHDR' 16 13
    2023-10-16 02:07:31,481 - STREAM b'sBIT' 41 3
    2023-10-16 02:07:31,481 - b'sBIT' 41 3 (unknown)
    2023-10-16 02:07:31,481 - STREAM b'IDAT' 56 8192
'''