class Edgel:

    def __init__(self, pos, orient):
        self.__position = pos
        self.__orientation = orient

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, pos):
        self.__position = pos

    @property
    def orientation(self):
        return self.__orientation

    @orientation.setter
    def orientation(self, orient):
        self.__orientation = orient
