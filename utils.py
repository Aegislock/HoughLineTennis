import cv2
import numpy as np

def whiten_image(img_path):
    image = cv2.imread(img_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 150, 255])

    mask = cv2.inRange(hsv_image, lower_white, upper_white)

    result_image = cv2.bitwise_and(image, image, mask=mask)

    return result_image


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresholded, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel)
    return dilated


def create_black(image):
    height, width, _ = image.shape
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    return black_image


def find_intersections(horizontal_lines, vertical_lines):
    intersections = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            rho_h = h_line[0]
            theta_h = h_line[1]
            rho_v = v_line[0]
            theta_v = v_line[1]
            A = np.array(
                [[np.cos(theta_h), np.sin(theta_h)], [np.cos(theta_v), np.sin(theta_v)]]
            )
            b = np.array([[rho_h], [rho_v]])
            intersection = np.linalg.solve(A, b)
            x, y = map(int, intersection)
            intersections.append((x, y))
    return intersections


def merge_similar_lines(lines, rho_d, theta_d):
    merged_lines = []

    while lines:
        base_line = lines.pop(0)
        rho_a, theta_a = base_line
        similar_lines = [base_line]

        lines_to_remove = []

        for line in lines:
            rho_b, theta_b = line
            if abs(rho_a - rho_b) <= rho_d and abs(theta_a - theta_b) <= theta_d:
                similar_lines.append(line)
                lines_to_remove.append(line)

        for line in lines_to_remove:
            lines.remove(line)

        rho_cumulative = sum(line[0] for line in similar_lines)
        theta_cumulative = sum(line[1] for line in similar_lines)
        rho_average = rho_cumulative / len(similar_lines)
        theta_average = theta_cumulative / len(similar_lines)

        merged_lines.append([rho_average, theta_average])

    return merged_lines


def hash_coordinates(array):
    hash = {}
    for x, y in array:
        if y not in hash:
            hash[y] = [x]
        else:
            hash[y].append(x)
    return hash


def construct_points(hash):
    top_line = min(hash.keys())
    bottom_line = max(hash.keys())
    top_left = (min(hash[top_line]), top_line)
    top_right = (max(hash[top_line]), top_line)
    bottom_left = (min(hash[bottom_line]), bottom_line)
    bottom_right = (max(hash[bottom_line]), bottom_line)
    return top_left, top_right, bottom_left, bottom_right


def map_lines(M, lines):
    transformed_lines = []

    for line in lines:
        pt1 = np.array([[line[0][0]], [line[0][1]], [1]], dtype=np.float32)
        pt2 = np.array([[line[1][0]], [line[1][1]], [1]], dtype=np.float32)

        transformed_pt1_homogeneous = M @ pt1
        transformed_pt2_homogeneous = M @ pt2

        transformed_pt1 = (
            transformed_pt1_homogeneous[:2] / transformed_pt1_homogeneous[2]
        )
        transformed_pt2 = (
            transformed_pt2_homogeneous[:2] / transformed_pt2_homogeneous[2]
        )

        transformed_lines.append((transformed_pt1, transformed_pt2))

    transformed_lines = np.array(transformed_lines, dtype=np.float32)

    return transformed_lines


def horizontal_overlaps(court_horizontal, reference_horizontal, threshold):
    overlaps = 0
    for ch in court_horizontal:
        _, y = ch[0]
        for rh in reference_horizontal:
            _, r_y = rh[0]
            if abs(y - r_y) < threshold:
                overlaps += 1
                break
    return overlaps


def vertical_overlaps(court_vertical, reference_vertical, threshold):
    overlaps = 0
    for ch in court_vertical:
        x, _ = ch[0]
        for rh in reference_vertical:
            r_x, _ = rh[0]
            if abs(x - r_x) < threshold:
                overlaps += 1
    return overlaps


def find_intersections_x_y(horizontal_lines, vertical_lines):
    intersections = []
    for h in horizontal_lines:
        y = h[0][1]
        for v in vertical_lines:
            x = v[0][0]
            intersection = [x, y]
            intersections.append(intersection)
    return intersections


def make_transparent(new_img):
    if new_img.shape[2] == 3:
        b, g, r = cv2.split(new_img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Fully opaque initially
        new_img = cv2.merge([b, g, r, alpha])

    black_condition = (
        (new_img[:, :, 0] == 0) & (new_img[:, :, 1] == 0) & (new_img[:, :, 2] == 0)
    )

    new_img[black_condition, 3] = 0


# Function to check if a line is vertical
def is_vertical(line, threshold=2):
    x1, y1, x2, y2 = line[0]
    return abs(x1 - x2) < threshold


# Function to find vertical lines near the center of the image
def vertical_lines_near_center(
    lines, image_width, image_height, h_margin=0.1, v_margin=0.3
):
    vertical_lines_near_center = []

    center_x = image_width / 2
    margin_x = image_width * h_margin

    center_y = image_height / 2
    margin_y = image_height * v_margin

    center_left = center_x - margin_x
    center_right = center_x + margin_x

    center_top = center_y - margin_y
    center_bottom = center_y + margin_y

    for line in lines:
        if (
            is_vertical(line)
            and (center_left <= line[0][0] <= center_right)
            and (
                center_top <= line[0][1] <= center_bottom
                and center_top <= line[0][3] <= center_bottom
            )
        ):
            vertical_lines_near_center.append(line)

    return vertical_lines_near_center


def transform_point(point, matrix):
    t_point = matrix @ np.array([[point[0]], [point[1]], [1]], dtype=np.float32)
    b_point = t_point[:2, 0] / t_point[2, 0]
    return b_point


def search_for_court(
    lowest_point,
    known_points,
    horizontal_lines,
    vertical_lines,
    h_trimmed,
    v_trimmed,
    tennis_court,
):
    best_score = 0
    best_matrices = []
    best_matrix = None
    thres = 5
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None
    for i in range(len(vertical_lines)):  # 4
        v_line_1 = vertical_lines[i]
        for j in range(i + 1, len(vertical_lines)):  # 3
            v_line_2 = vertical_lines[j]
            for k in range(len(horizontal_lines)):  # 4
                h_line_1 = horizontal_lines[k]
                for l in range(k + 1, len(horizontal_lines)):  # 3
                    h_line_2 = horizontal_lines[l]
                    # Find Intersections
                    h_lines = [h_line_1, h_line_2]
                    v_lines = [v_line_1, v_line_2]
                    intersections = find_intersections(h_lines, v_lines)  # 2 * 2 = 4
                    # Hash X at each Y level
                    hash_map = hash_coordinates(intersections)
                    top_left, top_right, bottom_left, bottom_right = construct_points(
                        hash_map
                    )  # PROCESSED IMAGE KEY POINTS
                    pts1 = np.array(
                        [
                            np.array(top_left),
                            np.array(top_right),
                            np.array(bottom_left),
                            np.array(bottom_right),
                        ],
                        dtype=np.float32,
                    )
                    # Establish Horizontal Lines
                    horizontal_rl = tennis_court.horizontal_lines
                    if v_line_1[1] > np.pi / 2 and v_line_2[1] > np.pi / 2:
                        # Search Right Side 6 Boxes
                        vertical_rl = (
                            tennis_court.vertical_lines[2],
                            tennis_court.vertical_lines[3],
                        )
                        for m in range(len(horizontal_rl)):
                            r_h_line_1 = horizontal_rl[m]
                            for n in range(m + 1, len(horizontal_rl)):
                                r_h_line_2 = horizontal_rl[n]
                                r_h_lines = [r_h_line_1, r_h_line_2]
                                r_v_lines = [vertical_rl[0], vertical_rl[1]]
                                r_intersections = find_intersections_x_y(
                                    r_h_lines, r_v_lines
                                )
                                r_hash_map = hash_coordinates(r_intersections)
                                (
                                    r_top_left,
                                    r_top_right,
                                    r_bottom_left,
                                    r_bottom_right,
                                ) = construct_points(
                                    r_hash_map
                                )  # REAL TENNIS COURT KEY POINTS
                                pts2 = np.array(
                                    [
                                        np.array(r_top_left),
                                        np.array(r_top_right),
                                        np.array(r_bottom_left),
                                        np.array(r_bottom_right),
                                    ],
                                    dtype=np.float32,
                                )
                                M = cv2.getPerspectiveTransform(pts1, pts2)
                                # Project Each Line with M
                                transformed_h_lines = map_lines(M, h_trimmed)
                                transformed_v_lines = map_lines(M, v_trimmed)
                                # Function to Count Hits and Misses
                                h_overlaps = min(
                                    horizontal_overlaps(
                                        tennis_court.horizontal_lines,
                                        transformed_h_lines,
                                        thres - 2,
                                    ),
                                    4,
                                )
                                v_overlaps = min(
                                    vertical_overlaps(
                                        tennis_court.vertical_lines,
                                        transformed_v_lines,
                                        thres,
                                    ),
                                    4,
                                )
                                score = h_overlaps + v_overlaps
                                if score == 8:
                                    return M, score
                                elif score > best_score:
                                    best_score = score  # THIS WAS NOT HERE
                                    best_matrices.clear()
                                    best_matrices.append(M)
                                elif score == best_score:
                                    best_matrices.append(M)
                    elif v_line_1[1] < np.pi / 2 and v_line_2[1] < np.pi / 2:
                        # Search Left Side 6 Boxes
                        vertical_rl = (
                            tennis_court.vertical_lines[0],
                            tennis_court.vertical_lines[1],
                        )
                        for m in range(len(horizontal_rl)):
                            r_h_line_1 = horizontal_rl[m]
                            for n in range(m + 1, len(horizontal_rl)):
                                r_h_line_2 = horizontal_rl[n]
                                r_h_lines = [r_h_line_1, r_h_line_2]
                                r_v_lines = [vertical_rl[0], vertical_rl[1]]
                                r_intersections = find_intersections_x_y(
                                    r_h_lines, r_v_lines
                                )
                                r_hash_map = hash_coordinates(r_intersections)
                                (
                                    r_top_left,
                                    r_top_right,
                                    r_bottom_left,
                                    r_bottom_right,
                                ) = construct_points(
                                    r_hash_map
                                )  # REAL TENNIS COURT KEY POINTS
                                pts2 = np.array(
                                    [
                                        np.array(r_top_left),
                                        np.array(r_top_right),
                                        np.array(r_bottom_left),
                                        np.array(r_bottom_right),
                                    ],
                                    dtype=np.float32,
                                )
                                M = cv2.getPerspectiveTransform(pts1, pts2)
                                # Project Each Line with M
                                transformed_h_lines = map_lines(M, h_trimmed)
                                transformed_v_lines = map_lines(M, v_trimmed)
                                # Function to Count Hits and Misses
                                h_overlaps = min(
                                    horizontal_overlaps(
                                        tennis_court.horizontal_lines,
                                        transformed_h_lines,
                                        thres,
                                    ),
                                    4,
                                )
                                v_overlaps = min(
                                    vertical_overlaps(
                                        tennis_court.vertical_lines,
                                        transformed_v_lines,
                                        thres,
                                    ),
                                    4,
                                )
                                score = h_overlaps + v_overlaps
                                if score == 8:
                                    return M, score
                                elif score > best_score:
                                    best_score = score  # THIS WAS NOT HERE
                                    best_matrices.clear()
                                    best_matrices.append(M)
                                elif score == best_score:
                                    best_matrices.append(M)
                    else:
                        # Search other 24 Boxes
                        vertical_rl_left = (
                            tennis_court.vertical_lines[0],
                            tennis_court.vertical_lines[1],
                        )
                        vertical_rl_right = (
                            tennis_court.vertical_lines[2],
                            tennis_court.vertical_lines[3],
                        )
                        for a in range(len(vertical_rl_left)):
                            left_line = vertical_rl_left[a]
                            for b in range(len(vertical_rl_right)):
                                right_line = vertical_rl_right[b]
                                for m in range(len(horizontal_rl)):
                                    r_h_line_1 = horizontal_rl[m]
                                    for n in range(m + 1, len(horizontal_rl)):
                                        r_h_line_2 = horizontal_rl[n]
                                        r_h_lines = [r_h_line_1, r_h_line_2]
                                        r_v_lines = [left_line, right_line]
                                        r_intersections = find_intersections_x_y(
                                            r_h_lines, r_v_lines
                                        )
                                        r_hash_map = hash_coordinates(r_intersections)
                                        (
                                            r_top_left,
                                            r_top_right,
                                            r_bottom_left,
                                            r_bottom_right,
                                        ) = construct_points(
                                            r_hash_map
                                        )  # REAL TENNIS COURT KEY POINTS
                                        pts2 = np.array(
                                            [
                                                np.array(r_top_left),
                                                np.array(r_top_right),
                                                np.array(r_bottom_left),
                                                np.array(r_bottom_right),
                                            ],
                                            dtype=np.float32,
                                        )
                                        M = cv2.getPerspectiveTransform(pts1, pts2)
                                        # Project Each Line with M
                                        transformed_h_lines = map_lines(M, h_trimmed)
                                        transformed_v_lines = map_lines(M, v_trimmed)
                                        # Function to Count Hits and Misses
                                        h_overlaps = min(
                                            horizontal_overlaps(
                                                tennis_court.horizontal_lines,
                                                transformed_h_lines,
                                                thres,
                                            ),
                                            4,
                                        )
                                        v_overlaps = min(
                                            vertical_overlaps(
                                                tennis_court.vertical_lines,
                                                transformed_v_lines,
                                                thres,
                                            ),
                                            4,
                                        )
                                        score = min(
                                            h_overlaps + v_overlaps,
                                            len(horizontal_lines) + len(vertical_lines),
                                        )
                                        if score == 8:
                                            return M, score
                                        elif score > best_score:
                                            best_score = score  # THIS WAS NOT HERE
                                            best_matrices.clear()
                                            best_matrices.append(M)
                                        elif score == best_score:
                                            best_matrices.append(M)
    # Tiebreaker Part 1
    if lowest_point is not None and known_points is not None:
        filtered = []
        for matrix in best_matrices:
            counter = 0
            inverse_matrix = np.linalg.inv(matrix)
            compare_array = [
                transform_point(tennis_court.point10, inverse_matrix),
                transform_point(tennis_court.point11, inverse_matrix),
                transform_point(tennis_court.point13, inverse_matrix),
                transform_point(tennis_court.point14, inverse_matrix),
            ]
            for point in known_points:
                for c_point in compare_array:
                    if np.linalg.norm(point - c_point) <= 10:
                        counter += 1
            if counter == len(known_points):
                filtered.append(matrix)
        closest = np.inf
        real_point = tennis_court.point12
        if len(filtered) != 0:
            for matrix in filtered:
                hypothetical_point = matrix @ np.array(
                    [[lowest_point[0]], [lowest_point[1]], [1]], dtype=np.float32
                )
                t_hypothetical_point = (
                    hypothetical_point[:2, 0] / hypothetical_point[2, 0]
                )
                distance = np.linalg.norm(t_hypothetical_point - real_point)
                if distance < closest:
                    closest = distance
                    best_matrix = matrix
        else:
            for matrix in best_matrices:
                hypothetical_point = matrix @ np.array(
                    [[lowest_point[0]], [lowest_point[1]], [1]], dtype=np.float32
                )
                t_hypothetical_point = (
                    hypothetical_point[:2, 0] / hypothetical_point[2, 0]
                )
                distance = np.linalg.norm(t_hypothetical_point - real_point)
                if distance < closest:
                    closest = distance
                    best_matrix = matrix

    return best_matrix, best_score
