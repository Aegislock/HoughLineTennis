import cv2
import os
import TennisCourtReference

from .utils import *

class TennisCourtDetector:
    def algorithm(img_path, whitened_image, output_path):
        base = cv2.imread(img_path)
        img = whitened_image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black = create_black(img)

        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
        lines = lsd.detect(gray)[0]
        drawn_img = np.copy(gray)  # Copy the original image to draw on
        drawn_img = cv2.cvtColor(drawn_img, cv2.COLOR_GRAY2BGR)  # Convert to color image to draw colored lines
        candidates = vertical_lines_near_center(lines, gray.shape[1], gray.shape[0])

        for line in candidates:
            x0, y0, x1, y1 = map(int, line[0])
            cv2.line(drawn_img, (x0, y0), (x1, y1), (255,0,0), 2)

        lowest_y = 0
        lowest_point = None
        for line in candidates:
            x0, y0, x1, y1 = map(int, line[0])
            if (y0 > y1):
                if (y0 > lowest_y):
                    lowest_y = y0
                    lowest_point = (x0, y0)
            else:
                if (y1 > lowest_y):
                    lowest_y = y1
                    lowest_point = (x1, y1)

        height, width = gray.shape
        max_dim = max(height, width)

        # Perform Canny edge detection
        edges = cv2.Canny(img, 25, 100, apertureSize=3, L2gradient=True)

        # Perform Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        horizontal_lines = []
        vertical_lines = []

        for rho, theta in lines[:, 0]:
            if theta < np.pi / 4 or theta > 3 * np.pi / 4:  # Vertical lines
                vertical_lines.append((rho, theta))
            elif np.pi / 4 <= theta <= 3 * np.pi / 4:  # Horizontal lines
                if np.pi/2 - 0.01 <= theta <= np.pi/2 + 0.01:
                    horizontal_lines.append((rho, theta))

        # Sort the lines by their rho value (distance from origin)
        horizontal_lines.sort(key=lambda x: x[0])
        vertical_lines.sort(key=lambda x: x[0])

        horizontal_lines = merge_similar_lines(horizontal_lines, 5, 0.01)
        vertical_lines = merge_similar_lines(vertical_lines, 20, 0.05)

        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return 

        intersections = find_intersections(horizontal_lines, vertical_lines)
        intersections.sort(key=lambda y: y[1])  

        hash = {}
        for x, y in intersections:
            if y not in hash:
                hash[y] = [x]
            else:
                hash[y].append(x)

        y_values = [y for y in hash]
        y_values.sort()

        h_trimmed = []
        v_trimmed = []

        for y in y_values:
            cv2.line(black, (min(hash[y]), y), (max(hash[y]), y), (0, 255, 0), 2)
            line = [[min(hash[y]), y], [max(hash[y]), y]]
            h_trimmed.append(line)

        #Draw Remaining Four (or Less) Lines
        top = y_values[0]
        bottom = y_values[len(y_values) - 1]

        cv2.line(black, (hash[top][0], top), (hash[bottom][0], bottom), (0, 255, 0), 2)
        if [[hash[top][0], top], [hash[bottom][0], bottom]] not in v_trimmed:
            v_trimmed.append([[hash[top][0], top], [hash[bottom][0], bottom]])

        cv2.line(black, (hash[top][1], top), (hash[bottom][1], bottom), (0, 255, 0), 2)
        if [[hash[top][1], top], [hash[bottom][1], bottom]] not in v_trimmed:
            v_trimmed.append([[hash[top][1], top], [hash[bottom][1], bottom]])

        cv2.line(black, (hash[top][len(hash[top]) - 2], top), (hash[bottom][len(hash[bottom]) - 2], bottom), (0, 255, 0), 2)
        if [[hash[top][len(hash[top]) - 2], top], [hash[bottom][len(hash[bottom]) - 2], bottom]] not in v_trimmed:
            v_trimmed.append([[hash[top][len(hash[top]) - 2], top], [hash[bottom][len(hash[bottom]) - 2], bottom]])

        cv2.line(black, (hash[top][len(hash[top]) - 1], top), (hash[bottom][len(hash[bottom]) - 1], bottom), (0, 255, 0), 2)
        if [[hash[top][len(hash[top]) - 1], top], [hash[bottom][len(hash[bottom]) - 1], bottom]] not in v_trimmed:
            v_trimmed.append([[hash[top][len(hash[top]) - 1], top], [hash[bottom][len(hash[bottom]) - 1], bottom]])

        for intersection in intersections:
            cv2.circle(black, intersection, 5, (255, 0, 0), -1)

        known_line = None
        known_points = None

        if lowest_point is not None:
            known_line = [(lowest_point[1], np.pi/2)]
            known_points = find_intersections(known_line, vertical_lines)
        
        real_court = TennisCourtReference('court_reference_background\\blank_black.png')

        matrix, score = search_for_court(lowest_point, known_points, horizontal_lines, vertical_lines, h_trimmed, v_trimmed, real_court)
        
        if matrix is None:
            return 

        inverse = np.linalg.inv(matrix)

        img1 = cv2.imread('court_reference_background\\court_lines_green.png')
        img2 = img
        rows, cols, _ = img2.shape
        new_img = cv2.warpPerspective(img1, inverse, (cols, rows))

        if new_img.shape[2] == 3:
            b, g, r = cv2.split(new_img)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Fully opaque initially
            new_img = cv2.merge([b, g, r, alpha])

        black_condition = (new_img[:, :, 0] == 0) & (new_img[:, :, 1] == 0) & (new_img[:, :, 2] == 0)

        new_img[black_condition, 3] = 0

        if img2.shape[2] == 3:
            b, g, r = cv2.split(img2)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Fully opaque initially
            img2 = cv2.merge([b, g, r, alpha])

        background = img2
        foreground = new_img

        foreground_rgb = foreground[..., :4]
        alpha_mask = foreground[..., 3]

        if foreground.shape[:2] != background.shape[:2]:
            foreground_rgb = cv2.resize(foreground_rgb, (background.shape[1], background.shape[0]))

        alpha_mask = alpha_mask / 255.0  
        alpha_mask = alpha_mask[..., np.newaxis]  

        composite_image = background * (1 - alpha_mask) + foreground_rgb * alpha_mask

        composite_image = composite_image.astype(np.uint8)

        cv2.imwrite(output_path, composite_image)