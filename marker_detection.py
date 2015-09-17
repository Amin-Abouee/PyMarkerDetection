import numpy as np
import cv2
import multiprocessing

class MarkerDetection:

	def __init__(self, image, grid_size):
		self.__image = image
		self.__GRID_SIZE = grid_size

	def image_division(self):
		image_rows, image_cols = self.__image.shape[:2]
		print self.__image.shape[:2]
		self.__grid_indices = [np.array([y, x]) for x in xrange (0, 
			image_cols, self.__GRID_SIZE) for y in xrange(0, image_rows, self.__GRID_SIZE)]

		# print len(self.__grid_indices)
		self.__pool = multiprocessing.Pool()

	@property
	def image(self):
	    return self.__image

	@image.setter
	def image(self, image):
		self.__image = image
	
	