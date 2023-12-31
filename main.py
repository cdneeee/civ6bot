import cv2 as cv
import numpy as np
import os
import database
from time import time
from screen_capt import WindowCapture
from vision import findClickPositions

# initialize the WindowCapture class
wincap = WindowCapture("Sid Meier's Civilization VI (DX11)")

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    findClickPositions(database.stone, screenshot,threshold=0.5, debug_mode='points')

    #cv.imshow('Computer Vision', screenshot)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')