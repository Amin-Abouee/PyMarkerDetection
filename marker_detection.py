import numpy as np
from numpy.polynomial import polynomial as poly
import cv2
import math
from pathos.multiprocessing import ProcessingPool as Pool
from edgel import Edgel, Line
from ransac import RansacLine


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
        self.__lines = []

    def subpix_sample(self, subpixel, frame):
        fx = int(subpixel[0])
        fy = int(subpixel[1])
        px = subpixel[0] - fx
        py = subpixel[1] - fy
        pixelValue = (1 - py) * (1 - px) * frame[fy, fx] + (1 - py) * px * frame[fy, fx + 1] + py * (1 - px) * frame[fy+1, fx] + py * px * frame[fy + 1, fx + 1]
        return pixelValue

    def subpixel_suppression(self, pos, gradient):
        precies_point = pos
        ix = self.__sobel_x[pos[0], pos[1]]
        iy = self.__sobel_y[pos[0], pos[1]]
        ix_norm = ix / gradient
        iy_norm = iy / gradient
        ix_norm_per = -iy / gradient
        iy_norm_per = ix / gradient
        for j in xrange(-1, 2):
            for i in xrange(-3, 4):
                subpixel_x = pos[0] + ix_norm * i + ix_norm_per * j
                subpixel_y = pos[1] + iy_norm * i + iy_norm_per * j
                if subpixel_x < self.__sobel_x.shape[1]-1 and subpixel_y < self.__sobel_x.shape[0]-1 and subpixel_x >= 0 and subpixel_y >= 0:
                    subpixel = np.array([subpixel_x, subpixel_y])
                    ix_new = self.subpix_sample(subpixel, self.__sobel_x)
                    iy_new = self.subpix_sample(subpixel, self.__sobel_y)
                    new_gradient = np.sqrt(ix_new ** 2 + iy_new ** 2)
                    if new_gradient > gradient:
                        gradient = new_gradient
                        precies_point = subpixel
        return precies_point

    def detect_edgels(self, scanline_positions):
        edgels_list = []
        for pos in scanline_positions:
            ix = self.__sobel_x[pos[0], pos[1]]
            iy = self.__sobel_y[pos[0], pos[1]]
            gradient = np.sqrt(ix ** 2 + iy ** 2)
            if gradient > self.__THRESHOLD:
                # res = self.subpixel_suppression(pos, gradient)
                # int_x = self.subpix_sample(res, self.__sobel_x)
                # int_y = self.subpix_sample(res, self.__sobel_y)
                # precies_point = np.array([int_x, int_y], dtype=int)
                # print pos, gradient, "->", precies_point, np.sqrt(
                # precies_point[0] ** 2 + precies_point[1] ** 2)
                # angle = math.atan2(precies_point[0], precies_point[1])
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
        return edgels

    def image_division(self):
        image_rows, image_cols = self.__image.shape[:2]
        print self.__image.shape[:2]
        grid_indices = [np.array([x, y]) for x in xrange(0,
            image_cols - self.__GRID_SIZE, self.__GRID_SIZE) for y in xrange(0,
            image_rows - self.__GRID_SIZE, self.__GRID_SIZE)]
        pool = Pool()
        output = pool.map(self.grid_division, grid_indices)
        threshod_sucess_sample = 6
        ransacGrouper = RansacLine(1, threshod_sucess_sample, 25, 2)
        for index, edgels in enumerate(output):
            if len(edgels) > threshod_sucess_sample:
                ransacGrouper.edgels = edgels
                ransac_groups = ransacGrouper.applay_parallel_ransac()
                self.line_segment(ransac_groups)

        # print len(self.__lines)
        # for line in self.__lines:
        #     print (line.slope, line.intercept)
        #     coefficients = np.array([line.slope, line.intercept])
        #     # print "cof: ", coefficients
        #     x = np.array([20, 50], dtype=np.int32)
        #     polynomial = np.poly1d(coefficients)
        #     # print "Poly: ", polynomial
        #     y = polynomial(x)
        #     y = [int(e) for e in y]
        #     print "x: ", x, "y: ", y
        #     cv2.line(self.__image, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 1)

        cv2.imshow('image', self.__image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print output
        # for e in output:
        #     for p in e:
        #         point = p.position
        #         cv2.circle(
        #         self.__image, (point[1], point[0]), 1, (255, 0, 0), -1)
        # cv2.imshow('image', self.__image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def line_segment(self, ransac_groups):
        points = []
        max_size = 0
        idx_max = 0
        for idx, value in enumerate(ransac_groups):
            if len(value) > max_size:
                idx_max = idx
                max_size = len(value)

        print "Index: ", idx_max, "Size: ", max_size
        if max_size > 0:
            best_edgels_group = ransac_groups[idx_max]
            for edgel in best_edgels_group:
                points.append(edgel.position)
            print points
            y = [pos.position[0] for pos in best_edgels_group]
            x = [pos.position[1] for pos in best_edgels_group]
            # for xp, yp in zip(x, y):
                # cv2.circle(self.__image, (yp, xp), 1, (255, 0, 0), -1)
            print "X: ", x
            print "Y: ", y
            # coefficients = poly.polyfit(x, y, 1)
            # print "Cof: ", coefficients
            # polynomial = np.poly1d(coefficients)
            # print "Poly: ", polynomial
            # print "Y: new", polynomial(x)
            vy, vx, cy, cx = cv2.fitLine(np.float32(points), cv2.DIST_L2, 0, 0.01, 0.01)
            cv2.line(self.__image, (int(cx-vx*50), int(cy-vy*50)), (int(cx+vx*50), int(cy+vy*50)), (0, 255, 255))
            # self.__lines.append(Line(coefficients[0], coefficients[1]))
            # cv2.line(self.__image, (y[0], x[0]), (y[1], x[1]), (0, 255, 0), 1)

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, image):
        self.__image = image
