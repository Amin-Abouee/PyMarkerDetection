import numpy as np
import cv2
import math
from pathos.multiprocessing import ProcessingPool as Pool
from edgel import Edgel


class MarkerDetection:

    def __init__(self, image, grid_size, baselines_size, threshold):
        self.__image = image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.__sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5,
            scale=0.0625)
        self.__sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5,
            scale=0.0625)
        self.__GRID_SIZE = grid_size
        self.__BASELINES_SIZE = baselines_size
        self.__THRESHOLD = threshold
        self.__edgels = []

    def detect_edgels(self, scanline_positions):
        edgels_list = []
        for pos in scanline_positions:
            ix = self.__sobel_x[pos[0], pos[1]]
            iy = self.__sobel_y[pos[0], pos[1]]
            if np.sqrt(ix ** 2 + iy ** 2) > self.__THRESHOLD:
                angle = math.atan2(iy, ix)
                edgels_list.append(Edgel(pos, angle))
        return edgels_list

    def grid_division(self, pos):
        # include the border
        bound_row = pos[0] + self.__GRID_SIZE + 1
        bound_col = pos[1] + self.__GRID_SIZE + 1
        scanline_positions = [np.array([x, y], dtype=np.float32) for x in
         xrange(pos[1], bound_col, self.__BASELINES_SIZE) for y in
          xrange(pos[0], bound_row, self.__BASELINES_SIZE)]
        edgels = self.detect_edgels(scanline_positions)
        print "Size: ", len(edgels)
        return edgels

    def image_division(self):
        image_rows, image_cols = self.__image.shape[:2]
        print self.__image.shape[:2]
        grid_indices = [np.array([x, y]) for x in xrange(0,
            image_cols - self.__GRID_SIZE, self.__GRID_SIZE) for y in xrange(0,
            image_rows - self.__GRID_SIZE, self.__GRID_SIZE)]
        self.__pool = Pool()
        output = self.__pool.map(self.grid_division, grid_indices)

        # print output
        for e in output:
            for p in e:
                point = p.position
                cv2.circle(self.__image, (point[1], point[0]), 1, (255, 0, 0), -1)
        cv2.imshow('image', self.__image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, image):
        self.__image = image
