import numpy as np
import cv2
import multiprocessing

class MarkerDetection:

    def __init__(self, image, grid_size, baselines_size):
        self.__image = image
        self.__sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5, scale=0.0625)
        self.__sobei_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5, scale=0.0625)
        self.__GRID_SIZE = grid_size
        self.__BASELINES_SIZE = baselines_size

    def image_division(self):
        image_rows, image_cols = self.__image.shape[:2]
        print self.__image.shape[:2]
        self.__grid_indices = [np.array([x, y]) for x in xrange (0, 
            image_cols - self.__GRID_SIZE, self.__GRID_SIZE) for y in xrange(0, image_rows - self.__GRID_SIZE, self.__GRID_SIZE)]
        print self.__grid_indices
        self.__pool = multiprocessing.Pool()


    @property
    def image(self):
        return self.__image

    @property
    def grid_indices(self):
        return self.__grid_indices
    
    @image.setter
    def image(self, image):
        self.__image = image
    
    