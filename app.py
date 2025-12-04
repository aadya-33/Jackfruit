import cv2 as cv
import numpy as np
import os

# detect red corners
def get_red_corners(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 

    # Red color ranges
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv.inRange(hsv, lower_red1, upper_red1) + cv.inRange(hsv, lower_red2, upper_red2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    corners = []
    for cnt in contours:
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            corners.append([cx, cy])

    if len(corners) != 4:
        raise ValueError(f"Expected 4 red corners, found {len(corners)}")
    
    return np.array(corners, dtype=np.float32)

# detect four corners in the hard copy
def get_paper_corners(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in hard copy")
    cnt = max(contours, key=cv.contourArea)
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) != 4:
        raise ValueError(f"Expected 4 corners in hard copy, found {len(approx)}")
    
    return np.array([p[0] for p in approx], dtype=np.float32)

# top left, bottom right, top right and bottom left
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

# to extract the table from the hard copy 
def extract_table(img):
    # Convert to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Threshold
    th = cv.adaptiveThreshold(gray, 255,
                               cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY_INV, 15, 5)

    # Detect horizontal lines
    hor_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
    horizontal = cv.morphologyEx(th, cv.MORPH_OPEN, hor_kernel)

    # Detect vertical lines
    ver_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 40))
    vertical = cv.morphologyEx(th, cv.MORPH_OPEN, ver_kernel)

    # Combine lines to get table structure
    table_mask = cv.add(horizontal, vertical)

    # Find biggest contour (the table)
    cnts, _ = cv.findContours(table_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None  # No table detected

    table_cnt = max(cnts, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(table_cnt)

    # Return ONLY the table crop
    table_crop = img[y:y+h, x:x+w]
    return table_crop

# to extract cells from the table
def save_table_cells_to_folder(table_img, n_rows, col_widths, col_names, folder="cells"):
    """
    table_img  : input table image
    n_rows     : number of equal rows
    col_widths : list of column widths in pixels, e.g., [45,370,130,130]
    col_names  : list of column names, e.g., ["slno","particulars","rate","quantity"]
    folder     : folder to save cells
    """
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    height, width = table_img.shape[:2]
    row_height = height // n_rows  # equal row heights

    for r in range(n_rows):
        y1 = r * row_height
        y2 = (r + 1) * row_height if r < n_rows - 1 else height
        row_img = table_img[y1:y2, :]

        x_start = 0
        for c, w in enumerate(col_widths):
            x1 = x_start
            x2 = x_start + w
            cell_img = row_img[:, x1:x2]
            if r == 0:
                continue

            filename = os.path.join(folder, f"{col_names[c]}{r}.jpg")
            cv.imwrite(filename, cell_img)

            x_start += w

# to stitch the cells of the particulars column into one
def stitch_column_vertically(folder, col_name, n_rows, output_name="particulars_stitched.jpg"):
    imgs = []
    for r in range(1, n_rows+1):  
        filename = os.path.join(folder, f"{col_name}{r}.jpg")
        if os.path.exists(filename):
            img = cv.imread(filename)
            imgs.append(img)
        else:
            print(f"Warning: {filename} not found!")

    if not imgs:
        print("No images to stitch!")
        return

    stitched = cv.vconcat(imgs)
    cv.imwrite(os.path.join(folder, output_name), stitched)
    print(f"Stitched image saved as {os.path.join(folder, output_name)}")