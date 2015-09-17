import numpy as np
import cv2
from pathos.multiprocessing import ProcessingPool as Pool


class MarkerDetection:

    def __init__(self, image, grid_size, baselines_size, threshold):
        self.__image = image
        self.__sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5, scale=0.0625)
        self.__sobei_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5, scale=0.0625)
        self.__GRID_SIZE = grid_size
        self.__BASELINES_SIZE = baselines_size
        self.__THRESHOLD = threshold

    def detect_edgle(self, pos):
        pass

    def grid_division(self, pos):
        # include the border
        bound_row = pos[0] + self.__GRID_SIZE + 1
        bound_col = pos[1] + self.__GRID_SIZE + 1
        baseline_positions = [np.array([x, y]) for x in xrange (pos[1], 
            bound_col, self.__BASELINES_SIZE) for y in xrange(pos[0], 
            bound_row, self.__BASELINES_SIZE)]

        return len(baseline_positions),

    def image_division(self):
        image_rows, image_cols = self.__image.shape[:2]
        print self.__image.shape[:2]
        grid_indices = [np.array([x, y]) for x in xrange (0, 
            image_cols - self.__GRID_SIZE, self.__GRID_SIZE) for y in xrange(0, 
            image_rows - self.__GRID_SIZE, self.__GRID_SIZE)]

        self.__pool = Pool()
        output = self.__pool.map(self.grid_division, grid_indices)
        print output

    @property
    def image(self):
        return self.__image

    
    @image.setter
    def image(self, image):
        self.__image = image
    
    