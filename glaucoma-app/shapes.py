from math import sqrt


class Point:
    ''' Geometric point '''

    def __init__(self, x=0, y=0):
        self.x, self.y = x, y

    def set(self, point):
        ''' Set the point by another point '''
        self.x, self.y = point.x, point.y

    def value(self):
        ''' Get value of the point '''
        return (self.x, self.y)

    def length2(self, point):
        ''' Get length squared between this point and another '''
        return pow(self.x - point.x, 2) + pow(self.y - point.y, 2)

    def scale(self, factor):
        ''' Tranforms the point (scale) '''
        self.x = round(self.x * factor)
        self.y = round(self.y * factor)

    def move(self, xdiff, ydiff):
        ''' Tranforms the point (move) '''
        self.x += xdiff
        self.y += ydiff

    def isIn(self, x, y, length):
        ''' Make sure you are inside the point square area
            The square side length is = 2 x length
            This is faster than using sqrt in calculations.
        '''
        if self.x + length >= x >= self.x - length and \
           self.y + length >= y >= self.y - length:
            return True
        else:
            return False


class Circle:
    ''' Geometric circle '''

    def __init__(self, cx=0, cy=0, rx=0, ry=0, condition=False):
        self.c, self.r = Point(cx, cy), Point(rx, ry)
        self.used = condition

    def __str__(self):
        ''' Print internal variables (used for debugging) '''
        cx, cy, rx, ry = self.c.value(), self.r.value()
        return f'Circle({cx}, {cy}, {rx}, {ry})'

    def copy(self):
        ''' Create a copy of the circle '''
        circle = Circle()
        circle.setCenter(self.c)
        circle.setRaduis(self.r)
        return circle

    def center(self):
        ''' Get the center point of the circle '''
        return self.c.value()

    def radius(self):
        ''' Get the radius point of the circle '''
        return self.r.value()

    def setCenter(self, point):
        ''' Set the center point of the circle '''
        self.c.set(point)

    def setRaduis(self, point):
        ''' Set the radius point of the circle '''
        self.r.set(point)

    def dia(self):
        ''' Calculate circle diameter '''
        radius = sqrt(self.c.length2(self.r))
        return round(2 * radius)

    def scale(self, factor):
        ''' Tranforms the circle (scale) '''
        self.c.scale(factor)
        self.r.scale(factor)

    def move(self, xdiff, ydiff):
        ''' Tranforms the circle (move) '''
        self.c.move(xdiff, ydiff)
        self.r.move(xdiff, ydiff)


class Ellipse:
    ''' Geometric Ellipse
        TODO: Not implemented yet.
    '''
    def __init__(self, cx=0, cy=0, px=0, py=0, qx=0, qy=0):
        self.c, self.i, self.j = Point(cx, cy), Point(px, py), Point(qx, qy)

    def __str__(self):
        ''' Print internal variables (used for debugging) '''
        cx, cy, ix, iy, jx, jy = self.c.value(), self.i.value(), self.j.value()
        return f'Ellipse({cx}, {cy}, {ix}, {iy}, {jx}, {jy})'
