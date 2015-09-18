import abc
import random
import math
from abc import ABCMeta
from abc import abstractmethod
from pathos.multiprocessing import ProcessingPool as Pool


class Ransac:
    __metaclass__ = ABCMeta

    def __init__(self, threshold_distance, threshold_sucess_sample, num_run):
        self._threshold_distance = threshold_distance
        self._num_run = num_run
        self._threshold_sucess_sample = threshold_sucess_sample

    @abstractmethod
    def calculate_distance(self, *args):
        return NotImplementedError

    @property
    def threshold(self):
        return self._threshold_distance

    @threshold.setter
    def threshold(self, threshold_distance):
        self._threshold_distance = threshold_distance

    @property
    def num_run(self):
        return self._num_run

    @num_run.setter
    def num_run(self, num_run):
        self._num_run = num_run

    @property
    def threshold_sucess_sample(self):
        return self._threshold_sucess_sample

    @threshold_sucess_sample.setter
    def threshold_sucess_sample(self, threshold_sucess_sample):
        self._threshold_sucess_sample = threshold_sucess_sample


class RansacLine(Ransac):

    def __init__(
        self, threshold_distance, threshold_sucess_sample, num_run, num_sample, edgels=None):
        super(RansacLine, self).__init__(threshold_distance, threshold_sucess_sample, num_run)
        self.__num_sample = num_sample
        self.__edgels = edgels

    def calculate_distance(self, *args):
        print "Attemp: ", args,
        cnt_success = 0
        sucess_edgels = []
        p1 = random.choice(self.__edgels)
        p2 = random.choice(self.__edgels)

        p1_x = p1.position[0]
        p1_y = p1.position[1]

        p2_x = p2.position[0]
        p2_y = p2.position[1]

        a = p2_y - p1_y
        b = p1_x - p2_x
        c = p2_x * p1_y - p1_x * p2_y

        for edgel in self.__edgels:
            p3 = edgel.position
            if math.sqrt(a ** 2 + b ** 2) > 0:
                distance = math.fabs(a * p3[0] + b * p3[1] + c) / math.sqrt(
                    a ** 2 + b ** 2)
                if distance < self._threshold_distance:
                    cnt_success += 1
                    sucess_edgels.append(edgel)

        if len(sucess_edgels) > self._threshold_sucess_sample:
            print len(sucess_edgels)
            return sucess_edgels
        else:
            print "0"
            None

    def applay_parallel_ransac(self):
        sample_indices = [i for i in xrange(25)]
        pool = Pool()
        output = pool.map(self.calculate_distance, sample_indices)
        return output

    @property
    def num_sample(self):
        return self.__num_sample

    @num_sample.setter
    def num_sample(self, num_sample):
        self.__num_sample = num_sample

    @property
    def edgels(self):
        return self.__edgels

    @edgels.setter
    def edgels(self, edgels):
        self.__edgels = edgels
        print "Size Edgels: ", len(self.__edgels)
