import cv2 as cv
import numpy as np
import os
from time import time
from windowCapture import WindowCapture
from Thresholding_v3 import findClickPositions

# initialize the WindowCapture class
wincap = WindowCapture(None)

loop_time = time()
while(True):
    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    #cv.imshow('Computer Vision', screenshot)
    findClickPositions('blob.jpg', screenshot, 0.70, 'rectangles')
    #findClickPositions('blob.jpg', screenshot, 0.43, 'rectangles')


    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')