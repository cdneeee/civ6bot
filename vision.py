import cv2 as cv
import numpy as np
import database

def findClickPositions(needle_img_path, haystack_img, threshold = 0.65, debug_mode=None):
   # haystack_img = cv.imread('game_screen.png',cv.IMREAD_UNCHANGED)
    needle_img = cv.imread(database.stone,cv.IMREAD_UNCHANGED)

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # There are 6 methods to choose from:
    # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
    method = cv.TM_CCOEFF_NORMED

    res = cv.matchTemplate(haystack_img,needle_img,method)

    # You can view the result of matchTemplate() like this:
    #cv.imshow('Res', res)
   # cv.waitKey()
    # If you want to save this result to a file, you'll need to normalize the result array
    # from 0..1 to 0..255, see:
    # https://stackoverflow.com/questions/35719480/opencv-black-image-after-matchtemplate
    #cv.imwrite('result_CCOEFF_NORMED.jpg', result * 255)


    locations = np.where(res >= threshold)
    # We can zip those up into a list of (x, y) position tuples
    locations = list(zip(*locations[::-1]))
    #print(locations)
    # You'll notice a lot of overlapping rectangles get drawn. We can eliminate those redundant
        # locations by using groupRectangles().
        # First we need to create the list of [x, y, w, h] rectangles
    rectangles = []
    for loc in locations:
        rect=[int(loc[0]), int(loc[1]), needle_w, needle_h]
        rectangles.append(rect)
        rectangles.append(rect)

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
    points = []
    if len(rectangles):
        print('Found needle.')

        line_color = (0, 255, 0)
        line_type = cv.LINE_4
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS

        # Loop over all the locations and draw their rectangle
        for (x, y, w, h) in rectangles:

            # Determine the center position
            center_x = x + int(w / 2)
            center_y = y + int(h / 2)
            # Save the points
            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                # Determine the box position
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                # Draw the box
                cv.rectangle(haystack_img, top_left, bottom_right, color=line_color,
                             lineType=line_type, thickness=2)
            elif debug_mode == 'points':
                # Draw the center point
                cv.drawMarker(haystack_img, (center_x, center_y),
                              color=marker_color, markerType=marker_type,
                              markerSize=25, thickness=1)

    if debug_mode:
        cv.imshow('Matches', haystack_img)
        # cv.waitKey()
        # cv.imwrite('result_click_point.jpg', haystack_img)

        return points