import cv2 as cv
import database
from time import time
from screen_capt import WindowCapture
from vision import Vision
from brains import *

# initialize the WindowCapture class
wincap = WindowCapture("Sid Meier's Civilization VI (DX11)")
vision_ = Vision(database.stone)

loop_time = time()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    #points = vision_.find(screenshot, 0.65, 'points')
    # pre-process the image


    cv.imshow('Computer Vision', screenshot)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')