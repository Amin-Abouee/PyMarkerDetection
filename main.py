# import numpy as np
import cv2
import sys
from marker_detection import MarkerDetection


if __name__ == "__main__":
	img = cv2.imread(sys.argv[1], 1)
	mk = MarkerDetection(img, 5)
	mk.image_division()