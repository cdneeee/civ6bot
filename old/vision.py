import cv2 as cv
import numpy as np
import database
from edgeFilter import EdgeFilter
from hsvfilter import HsvFilter

class Vision:
    # constants
    TRACKBAR_WINDOW = "Trackbars"

    # properties
    needle_img = None
    needle_w = 0
    needle_h = 0
    method = None
    def __init__(self, needle_img_path, method=cv.TM_CCOEFF_NORMED):
        # load the image we're trying to match
        # https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html
        self.needle_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)

        # Save the dimensions of the needle image
        self.needle_w = self.needle_img.shape[1]
        self.needle_h = self.needle_img.shape[0]

        # There are 6 methods to choose from:
        # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
        self.method = method
    def findClickPositions(self, haystack_img, threshold = 0.65, debug_mode=None):
       # haystack_img = cv.imread('game_screen.png',cv.IMREAD_UNCHANGED)
       # needle_img = cv.imread(database.stone,cv.IMREAD_UNCHANGED)

        # There are 6 methods to choose from:
        # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
        #method = cv.TM_CCOEFF_NORMED

        res = cv.matchTemplate(haystack_img,self.needle_img,self.method)

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
            rect=[int(loc[0]), int(loc[1]), self.needle_w,self.needle_h]
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


    def init_control_gui(self):
        cv.namedWindow(self.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass
        # OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
        cv.createTrackbar('HMin', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, 255)

        # trackbars for increasing/decreasing saturation and value
        cv.createTrackbar('SAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACKBAR_WINDOW, 0, 255, nothing)

        # trackbars for edge creation
        cv.createTrackbar('KernelSize', self.TRACKBAR_WINDOW, 1, 30, nothing)
        cv.createTrackbar('ErodeIter', self.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('DilateIter', self.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('Canny1', self.TRACKBAR_WINDOW, 0, 200, nothing)
        cv.createTrackbar('Canny2', self.TRACKBAR_WINDOW, 0, 500, nothing)
        # Set default value for Canny trackbars
        cv.setTrackbarPos('KernelSize', self.TRACKBAR_WINDOW, 5)
        cv.setTrackbarPos('Canny1', self.TRACKBAR_WINDOW, 100)
        cv.setTrackbarPos('Canny2', self.TRACKBAR_WINDOW, 200)

    def get_hsv_filter_from_controls(self):
        # Get current positions of all trackbars
        hsv_filter = HsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', self.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACKBAR_WINDOW)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACKBAR_WINDOW)
        return hsv_filter

    def get_edge_filter_from_controls(self):
        # Get current positions of all trackbars
        edge_filter = EdgeFilter()
        edge_filter.kernelSize = cv.getTrackbarPos('KernelSize', self.TRACKBAR_WINDOW)
        edge_filter.erodeIter = cv.getTrackbarPos('ErodeIter', self.TRACKBAR_WINDOW)
        edge_filter.dilateIter = cv.getTrackbarPos('DilateIter', self.TRACKBAR_WINDOW)
        edge_filter.canny1 = cv.getTrackbarPos('Canny1', self.TRACKBAR_WINDOW)
        edge_filter.canny2 = cv.getTrackbarPos('Canny2', self.TRACKBAR_WINDOW)
        return edge_filter

    def apply_hsv_filter(self, original_image, hsv_filter=None):
        # convert image to HSV
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

        # if we haven't been given a defined filter, use the filter values from the GUI
        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        # add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])
        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() to display it properly
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

    def apply_edge_filter(self, original_image, edge_filter=None):
        # if we haven't been given a defined filter, use the filter values from the GUI
        if not edge_filter:
            edge_filter = self.get_edge_filter_from_controls()

        kernel = np.ones((edge_filter.kernelSize, edge_filter.kernelSize), np.uint8)
        eroded_image = cv.erode(original_image, kernel, iterations=edge_filter.erodeIter)
        dilated_image = cv.dilate(eroded_image, kernel, iterations=edge_filter.dilateIter)

        # canny edge detection
        result = cv.Canny(dilated_image, edge_filter.canny1, edge_filter.canny2)

        # convert single channel image back to BGR
        img = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

        return img

    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c


