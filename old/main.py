import cv2 as cv
import numpy as np
import os
import database
from time import time
from screen_capt import WindowCapture
from vision import Vision
from hsvfilter import HsvFilter
from edgeFilter import EdgeFilter

# initialize the WindowCapture class
wincap = WindowCapture("Sid Meier's Civilization VI (DX11)")
vision_ = Vision('stone_resource.png')
# initialize the trackbar window
vision_.init_control_gui()

hsv_filter = HsvFilter(0, 180, 129, 15, 229, 243, 143, 0, 67, 0)
loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    #Vision.findClickPositions(database.stone, screenshot,threshold=0.5, debug_mode='points')
    # pre-process the image
    processed_image = vision_.apply_hsv_filter(screenshot)

    edges_image = vision_.apply_edge_filter(processed_image)

    #cv.imshow('Computer Vision', screenshot)
    cv.imshow('Processed', processed_image)
    cv.imshow('Edges', edges_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')