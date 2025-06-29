class TennisCourtReference:
    def __init__(self, blank_image_path):
        self.point1 = (0 + 60, 0 + 60)
        self.point2 = (45 + 60, 0 + 60)
        self.point3 = (315 + 60, 0 + 60)
        self.point4 = (360 + 60, 0 + 60)
        self.point5 = (0 + 60, 180 + 60)
        self.point6 = (45 + 60, 180 + 60)
        self.point7 = (180 + 60, 180 + 60)
        self.point8 = (315 + 60, 180 + 60)
        self.point9 = (360 + 60, 180 + 60)
        self.point10 = (0 + 60, 600 + 60)
        self.point11 = (45 + 60, 600 + 60)
        self.point12 = (180 + 60, 600 + 60)
        self.point13 = (315 + 60, 600 + 60)
        self.point14 = (360 + 60, 600 + 60)
        self.point15 = (0 + 60, 780 + 60)
        self.point16 = (45 + 60, 780 + 60)
        self.point17 = (315 + 60, 780 + 60)
        self.point18 = (360 + 60, 780 + 60)
        self.horizontal_lines = (
            ((self.point1), (self.point4)),
            ((self.point5), (self.point9)),
            ((self.point10), (self.point14)),
            ((self.point15), (self.point18)),
        )
        self.vertical_lines = (
            ((self.point1), (self.point15)),
            ((self.point2), (self.point16)),
            ((self.point3), (self.point17)),
            ((self.point4), (self.point18)),
        )
        self.blank_image = cv2.imread(blank_image_path)

    def draw_lines(self, thickness, circles=False, color=(255, 255, 255)):
        img = self.blank_image
        cv2.line(img, (self.point1), (self.point4), color, thickness)
        cv2.line(img, (self.point5), (self.point9), color, thickness)
        cv2.line(img, (self.point10), (self.point14), color, thickness)
        cv2.line(img, (self.point15), (self.point18), color, thickness)
        cv2.line(img, (self.point1), (self.point15), color, thickness)
        cv2.line(img, (self.point2), (self.point16), color, thickness)
        cv2.line(img, (self.point7), (self.point12), color, thickness)
        cv2.line(img, (self.point3), (self.point17), color, thickness)
        cv2.line(img, (self.point4), (self.point18), color, thickness)
        if circles:
            cv2.circle(img, self.point1, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point2, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point3, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point4, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point5, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point6, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point7, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point8, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point9, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point10, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point11, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point12, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point13, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point14, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point15, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point16, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point17, 5, (255, 0, 0), -1)
            cv2.circle(img, self.point18, 5, (255, 0, 0), -1)
        return img
