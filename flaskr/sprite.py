import math

class Sprite():
    def __init__(self, x, y, angle, status, battery):
        super().__init__()
        self.x = x
        self.y = y
        self.angle = angle

        self.speed = 1
        self.battery = 100
        self.status = status
        self.ontask = 0 
        self.recieved = False

        self.target_x = None
        self.target_y = None

        self.path = []
        self.i = 0

    
    def set_path(self, path):
        self.path = path
        self.path_length = len(path)
        self.i = 0

    def sprite_move(self, t):
        if self.path and self.i < self.path_length and self.path[self.i][2] <= t:
            self.move_to(self.path[self.i][0], self.path[self.i][1])
            self.i += int(self.speed)
            return self.speed, [float(self.x), float(self.y), float(self.angle)]
        elif self.path == []:
            return 0, [float(self.x),float(self.y), float(self.angle)]
        elif self.path and self.i >= self.path_length:
            self.i = 0
            self.path, self.openSet, self.closeSet = [], [], []
            self.recieved = True
            return 0, [self.x, self.y, self.angle]
        else:
            return 0, [self.x, self.y, self.angle]

    def move_to(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        self.angle = target_angle

        self.x, self.y = target_x, target_y
