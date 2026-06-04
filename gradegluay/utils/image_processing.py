from pathlib import Path

import cv2
import numpy as np


TEN_BAHT_COIN_DIAMETER_CM = 2.6
STANDARD_M_WIDTH_CM = 35
STANDARD_M_PRICE_BAHT = 30
COIN_ROI_SIZE_PX = 80
CLICK_REFINEMENT_RADIUS_PX = 35
COIN_DETECTION_ERROR_MESSAGE = (
    "ไม่สามารถตรวจจับเหรียญได้ กรุณาถ่ายรูปใหม่โดยวางเหรียญ 10 บาท"
    "ให้ชัดเจน และวางบนพื้นหลังสีเข้ม"
)


def refine_coin_click_center(image, center_x, center_y):
    image_height, image_width = image.shape[:2]
    x1 = max(0, center_x - CLICK_REFINEMENT_RADIUS_PX)
    y1 = max(0, center_y - CLICK_REFINEMENT_RADIUS_PX)
    x2 = min(image_width, center_x + CLICK_REFINEMENT_RADIUS_PX + 1)
    y2 = min(image_height, center_y + CLICK_REFINEMENT_RADIUS_PX + 1)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return center_x, center_y

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    search_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(
        search_mask,
        (center_x - x1, center_y - y1),
        CLICK_REFINEMENT_RADIUS_PX,
        255,
        -1,
    )
    gray = cv2.bitwise_and(gray, gray, mask=search_mask)
    _, bright_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    bright_mask = cv2.bitwise_and(bright_mask, search_mask)

    if cv2.countNonZero(bright_mask) == 0:
        _, _, _, max_location = cv2.minMaxLoc(gray, mask=search_mask)
        return x1 + max_location[0], y1 + max_location[1]

    moments = cv2.moments(bright_mask)
    if moments["m00"] == 0:
        return center_x, center_y

    refined_x = int(round(x1 + (moments["m10"] / moments["m00"])))
    refined_y = int(round(y1 + (moments["m01"] / moments["m00"])))

    return (
        int(np.clip(refined_x, 0, image_width - 1)),
        int(np.clip(refined_y, 0, image_height - 1)),
    )


def refine_coin_radius_with_hough(image, center_x, center_y, radius):
    if radius <= 0:
        return None

    image_height, image_width = image.shape[:2]
    crop_half_size = int(round(radius * 2.5))
    x1 = max(0, center_x - crop_half_size)
    y1 = max(0, center_y - crop_half_size)
    x2 = min(image_width, center_x + crop_half_size)
    y2 = min(image_height, center_y + crop_half_size)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    min_radius = max(1, int(radius * 0.6))
    max_radius = max(min_radius + 1, int(radius * 1.1))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, int(radius)),
        param1=100,
        param2=24,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None or len(circles[0]) == 0:
        return None

    local_center = np.array([center_x - x1, center_y - y1], dtype=np.float32)
    candidates = np.round(circles[0]).astype(int)
    best_circle = None
    best_distance = float("inf")

    for circle_x, circle_y, circle_radius in candidates:
        distance = float(np.linalg.norm(np.array([circle_x, circle_y]) - local_center))
        if distance <= radius * 0.5 and distance < best_distance:
            best_distance = distance
            best_circle = int(circle_radius)

    if best_circle is None:
        return None

    return min(best_circle, radius)


def parse_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing or invalid {field_name}. Please tap the center of the "
            "10-Baht coin."
        ) from exc


def calculate_grade_and_price(measured_width_cm):
    if measured_width_cm < 20:
        grade = "S"
        recommended_price = 20
    elif measured_width_cm < 23:
        grade = "S"
        recommended_price = 25
    elif measured_width_cm < 26:
        grade = "M"
        recommended_price = 30
    elif measured_width_cm < 29:
        grade = "M"
        recommended_price = 35
    else:
        grade = "L"
        recommended_price = 40

    return grade, recommended_price


def auto_detect_coin_from_click(image, center_x, center_y):
    image_height, image_width = image.shape[:2]
    half_roi = COIN_ROI_SIZE_PX // 2
    x1 = max(0, center_x - half_roi)
    y1 = max(0, center_y - half_roi)
    x2 = min(image_width, center_x + half_roi)
    y2 = min(image_height, center_y + half_roi)
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        raise ValueError(COIN_DETECTION_ERROR_MESSAGE)

    local_click = (center_x - x1, center_y - y1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    )

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_radius = None
    best_score = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        if radius < 12 or radius > half_roi:
            continue

        distance_to_click = np.hypot(circle_x - local_click[0], circle_y - local_click[1])
        if distance_to_click > radius * 0.85:
            continue

        circle_area = np.pi * radius * radius
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        fill_ratio = min(area / circle_area, 1.0)
        score = (
            (circularity * 0.65)
            + (fill_ratio * 0.25)
            + ((1 - distance_to_click / max(radius, 1)) * 0.10)
        )

        if circularity >= 0.35 and score > best_score:
            best_score = score
            best_radius = radius

    if best_radius is None:
        raise ValueError(COIN_DETECTION_ERROR_MESSAGE)

    return int(round(best_radius)), None


def detect_banana_comb(image, coin):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, np.array([15, 45, 45]), np.array([42, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
    banana_mask = cv2.bitwise_or(yellow_mask, green_mask)

    coin_mask = np.zeros(banana_mask.shape, dtype=np.uint8)
    cv2.circle(coin_mask, coin["center"], int(coin["radius"] * 1.35), 255, -1)
    banana_mask = cv2.bitwise_and(banana_mask, cv2.bitwise_not(coin_mask))

    kernel = np.ones((9, 9), dtype=np.uint8)
    banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > image.shape[0] * image.shape[1] * 0.01
    ]

    if not contours:
        return None, banana_mask

    merged_points = np.vstack(contours)
    hull = cv2.convexHull(merged_points)
    x, y, width, height = cv2.boundingRect(hull)

    return {
        "hull": hull,
        "bbox": (int(x), int(y), int(width), int(height)),
        "width_px": float(width),
    }, banana_mask


def process_banana_image(image_path, click_x_ratio, click_y_ratio, annotated_upload_dir):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read the uploaded image.")

    display_img = img.copy()

    orig_h, orig_w = img.shape[:2]
    image_height, image_width = orig_h, orig_w
    click_x_ratio = parse_float(click_x_ratio, "click_x_ratio")
    click_y_ratio = parse_float(click_y_ratio, "click_y_ratio")

    click_x_ratio = float(np.clip(click_x_ratio, 0, 1))
    click_y_ratio = float(np.clip(click_y_ratio, 0, 1))
    center_x = int(np.clip(click_x_ratio * orig_w, 0, orig_w - 1))
    center_y = int(np.clip(click_y_ratio * orig_h, 0, orig_h - 1))
    center_x, center_y = refine_coin_click_center(img, center_x, center_y)

    roi_w = int(round(max(80, orig_w * 0.10)))
    roi_h = int(round(max(80, orig_h * 0.10)))
    half_roi_w = roi_w // 2
    half_roi_h = roi_h // 2
    x1 = max(0, center_x - half_roi_w)
    y1 = max(0, center_y - half_roi_h)
    x2 = min(image_width, center_x + half_roi_w)
    y2 = min(image_height, center_y + half_roi_h)
    roi = img[y1:y2, x1:x2]
    local_click_x = center_x - x1
    local_click_y = center_y - y1
    actual_roi_h, actual_roi_w = roi.shape[:2]

    warning = None

    if roi.size == 0:
        print("Coin radius detection failed: ROI is empty or outside image bounds.")
        raise ValueError(COIN_DETECTION_ERROR_MESSAGE)
    else:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        threshold_maps = [
            cv2.adaptiveThreshold(
                roi_blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                2,
            ),
            cv2.adaptiveThreshold(
                roi_blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                21,
                2,
            ),
        ]

        best_radius = None
        best_center = None
        best_circularity = -1
        best_distance = float("inf")
        contour_count = 0
        roi_min_dimension = min(actual_roi_w, actual_roi_h)
        min_coin_radius = max(8, int(round(roi_min_dimension * 0.04)))
        max_coin_radius = max(min_coin_radius + 1, int(round(roi_min_dimension * 0.55)))
        max_click_distance = max(roi_min_dimension * 0.22, min_coin_radius * 1.5)

        for threshold_map in threshold_maps:
            threshold_map = cv2.morphologyEx(
                threshold_map,
                cv2.MORPH_CLOSE,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            contours, _ = cv2.findContours(
                threshold_map,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contour_count += len(contours)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 0:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue

                circularity = 4 * 3.14159 * area / (perimeter * perimeter)
                if circularity <= 0.45:
                    continue

                (circle_x, circle_y), candidate_radius = cv2.minEnclosingCircle(contour)
                distance_to_click = np.hypot(circle_x - local_click_x, circle_y - local_click_y)

                if candidate_radius < min_coin_radius or candidate_radius > max_coin_radius:
                    continue

                if distance_to_click > max(max_click_distance, candidate_radius * 1.25):
                    continue

                if circularity > best_circularity or (
                    np.isclose(circularity, best_circularity)
                    and distance_to_click < best_distance
                ):
                    best_circularity = circularity
                    best_distance = distance_to_click
                    best_radius = candidate_radius
                    best_center = (circle_x, circle_y)

        if best_radius is not None and best_center is not None:
            center_x = int(np.clip(round(x1 + best_center[0]), 0, image_width - 1))
            center_y = int(np.clip(round(y1 + best_center[1]), 0, image_height - 1))
            radius = int(round(best_radius))
        else:
            print(
                "Coin radius detection failed: adaptive threshold found "
                f"{contour_count} contours, but none had circularity > 0.45 "
                f"radius between {min_coin_radius}-{max_coin_radius} px, "
                "and center close enough to the tapped point."
            )
            raise ValueError(COIN_DETECTION_ERROR_MESSAGE)

    refined_radius = refine_coin_radius_with_hough(img, center_x, center_y, radius)
    if refined_radius is None:
        raise ValueError(COIN_DETECTION_ERROR_MESSAGE)

    radius = refined_radius

    coin = {
        "center": (center_x, center_y),
        "radius": radius,
        "diameter_px": float(radius * 2),
    }

    pixels_per_metric = coin["diameter_px"] / TEN_BAHT_COIN_DIAMETER_CM
    coin_radius_cm = (radius * 2) / pixels_per_metric
    if coin_radius_cm > 3.5 or coin_radius_cm < 1.8:
        raise ValueError(COIN_DETECTION_ERROR_MESSAGE)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    banana_mask = cv2.inRange(
        hsv,
        np.array([25, 40, 40], dtype=np.uint8),
        np.array([85, 255, 255], dtype=np.uint8),
    )

    coin_mask = np.zeros(banana_mask.shape, dtype=np.uint8)
    cv2.circle(coin_mask, coin["center"], int(coin["radius"] * 1.35), 255, -1)
    banana_mask = cv2.bitwise_and(banana_mask, cv2.bitwise_not(coin_mask))

    banana_kernel = np.ones((9, 9), dtype=np.uint8)
    banana_mask = cv2.morphologyEx(
        banana_mask,
        cv2.MORPH_OPEN,
        banana_kernel,
        iterations=1,
    )
    banana_mask = cv2.morphologyEx(
        banana_mask,
        cv2.MORPH_CLOSE,
        banana_kernel,
        iterations=3,
    )

    contours, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_banana_area = image_height * image_width * 0.01
    banana_contours = [
        contour for contour in contours if cv2.contourArea(contour) > min_banana_area
    ]

    if not banana_contours:
        raise ValueError("Could not segment the banana comb from the background.")

    banana_contour = max(banana_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(banana_contour)
    banana_hull = cv2.convexHull(banana_contour)
    x, y, width, height = cv2.boundingRect(banana_hull)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)
    center, size, angle = rect
    comb_width_pixels = max(size[0], size[1])
    comb_length_pixels = min(size[0], size[1])

    total_width_cm = comb_width_pixels / pixels_per_metric
    rounded_width_cm = round(float(total_width_cm), 2)
    comb_length_cm = round(float(comb_length_pixels / pixels_per_metric), 2)
    comb_area_cm2 = round(
        float(cv2.contourArea(banana_contour) / (pixels_per_metric ** 2)),
        2,
    )
    grade, recommended_price = calculate_grade_and_price(rounded_width_cm)
    cv2.circle(display_img, coin["center"], coin["radius"], (0, 0, 255), 3)
    cv2.circle(display_img, coin["center"], 5, (0, 0, 255), -1)
    cv2.putText(
        display_img,
        f"10 Baht coin: 2.6 cm ({radius}px radius)",
        (max(10, center_x - radius), max(28, center_y - radius - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.drawContours(display_img, [banana_hull], -1, (0, 180, 0), 3)
    cv2.rectangle(display_img, (x, y), (x + width, y + height), (0, 140, 255), 3)
    cv2.drawContours(display_img, [box], 0, (255, 0, 0), 2)

    cv2.putText(
        display_img,
        f"Width: {rounded_width_cm:.2f} cm | Grade: {grade} | Price: {recommended_price} Baht",
        (x, max(30, y - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 140, 255),
        2,
        cv2.LINE_AA,
    )

    if warning:
        cv2.putText(
            display_img,
            "Warning: default coin radius used",
            (20, min(image_height - 20, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    annotated_upload_dir = Path(annotated_upload_dir)
    annotated_filename = f"processed_{Path(image_path).stem}.jpg"
    annotated_path = annotated_upload_dir / annotated_filename

    try:
        annotated_upload_dir.mkdir(parents=True, exist_ok=True)
        saved = cv2.imwrite(str(annotated_path), display_img)
    except OSError as exc:
        raise RuntimeError("Could not save processed image.") from exc

    if not saved:
        raise RuntimeError("Could not save processed image.")

    return {
        "width": rounded_width_cm,
        "length": comb_length_cm,
        "area": comb_area_cm2,
        "grade": grade,
        "price": recommended_price,
        "annotated_image": f"uploads/{annotated_filename}",
        "warning": warning,
    }
