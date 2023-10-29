import os 
import pyautogui
import numpy as np
import cv2
import time
# im1 = pyautogui.screenshot()
# im1 = np.array(im1) #Load t/his data into our model for inference!


#Record screenshots during live play.
def record(num_seconds, head_start=10 ,delay=0.1):
    time.sleep(head_start) #Give me three seconds to start
    start_time = time.time()
    current_time = time.time()
    count = 0
    while current_time - start_time < num_seconds:
        # time.sleep(0.1)
        im1 = pyautogui.screenshot(os.getcwd() +  '/footage/' + str(count) +  '.png')
        im1 = np.array(im1) #Load this data into our model for inference!
        
        #Prints a bloody solid screenshot
        # print(im1.shape, type(im1))
        current_time = time.time()
        count+=1
        # time.sleep(delay)
#
if __name__ == "__main__":
    #15 min * 60s = 9000
    print('Thanks for starting!')
    record(9000, head_start=10, delay=0.5)
