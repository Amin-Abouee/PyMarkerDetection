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
		cols_scanline_positions = [np.array ([0, x], dtype = np.float32) for x in xrange (0, image_cols, self.__GRID_SIZE)]
		rows_scanline_positions = [np.array ([y, 0], dtype = np.float32) for y in xrange (self.__GRID_SIZE, image_rows, self.__GRID_SIZE)]
		self.__scanline_positions = cols_scanline_positions + rows_scanline_positions
		# print len(self.__scanline_positions)
		# print self.__scanline_positions
		self.__pool = multiprocessing.Pool()


	@property
	def image(self):
	    return self.__image

	@property
	def scanline_positions(self):
	    return self.__scanline_positions
	
	@image.setter
	def image(self, image):
		self.__image = image
	
	