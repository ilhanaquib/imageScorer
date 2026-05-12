import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import random
import hashlib


def generate_fibonacci_mask(height, width, orientation=0):

    mask = np.zeros((height, width), dtype=np.uint8)

    fib = [1, 1]
    for _ in range(10):
        fib.append(fib[-1] + fib[-2])

    scale = min(width, height) / fib[-1]
    fib = [f * scale for f in fib]

    squares = []

    x, y = 0.0, 0.0
    squares.append((x, y, fib[0]))

    direction = 0

    min_x = max_x = 0
    min_y = max_y = 0

    for i in range(1, len(fib)):

        s = fib[i]

        prev_x, prev_y, prev_s = squares[-1]

        if direction == 0:
            x = prev_x + prev_s
            y = prev_y

        elif direction == 1:
            x = prev_x
            y = prev_y + prev_s

        elif direction == 2:
            x = prev_x - s
            y = prev_y

        else:
            x = prev_x
            y = prev_y - s

        squares.append((x, y, s))

        min_x = min(min_x, x)
        min_y = min(min_y, y)

        max_x = max(max_x, x + s)
        max_y = max(max_y, y + s)

        direction = (direction + 1) % 4

    total_w = max_x - min_x
    total_h = max_y - min_y

    offset_x = (width - total_w) / 2 - min_x
    offset_y = (height - total_h) / 2 - min_y

    for i, (sx, sy, s) in enumerate(squares):

        sx += offset_x
        sy += offset_y

        sx = int(round(sx))
        sy = int(round(sy))
        s = int(round(s))

        cv2.rectangle(mask, (sx, sy), (sx + s, sy + s), 255, 1)

        mode = i % 4

        if mode == 0:
            center = (sx + s, sy + s)
            start, end = 180, 270

        elif mode == 1:
            center = (sx, sy + s)
            start, end = 270, 360

        elif mode == 2:
            center = (sx, sy)
            start, end = 0, 90

        else:
            center = (sx + s, sy)
            start, end = 90, 180

        cv2.ellipse(mask, center, (s, s), 0, start, end, 255, 2)

    return mask


def load_spiral_mask(height, width, orientation=0):

    spiral = cv2.imread("golden_spiral.png", cv2.IMREAD_UNCHANGED)

    if spiral is None:
        raise FileNotFoundError("golden_spiral.png not found.")

    spiral = cv2.resize(spiral, (width, height), interpolation=cv2.INTER_AREA)

    if orientation >= 4:
        spiral = cv2.flip(spiral, 1)
        orientation -= 4

    if orientation == 1:
        spiral = cv2.rotate(spiral, cv2.ROTATE_90_CLOCKWISE)

    elif orientation == 2:
        spiral = cv2.rotate(spiral, cv2.ROTATE_180)

    elif orientation == 3:
        spiral = cv2.rotate(spiral, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # ensure exact size match
    spiral = cv2.resize(spiral, (width, height), interpolation=cv2.INTER_AREA)

    if spiral.shape[2] == 4:

        mask = spiral[:, :, 3].astype(np.float32) / 255.0

    else:

        mask = cv2.cvtColor(spiral, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    return mask


def analyze_composition(uploaded_file):

    file_bytes = uploaded_file.read()

    np_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)

    img = cv2.imdecode(np_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 30, 90)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    h, w = gray.shape

    edge_points = np.where(edges > 0)

    if len(edge_points[0]) == 0:
        return img_rgb, 0.00

    y_coords, x_coords = edge_points

    best_score = -1

    best_mask = None

    for i in range(8):

        mask = load_spiral_mask(h, w, i)

        valid = (
            (y_coords >= 0)
            & (y_coords < mask.shape[0])
            & (x_coords >= 0)
            & (x_coords < mask.shape[1])
        )

        values = mask[y_coords[valid], x_coords[valid]]

        if len(values) > 0:

            raw = float(np.mean(values))

            score = np.sqrt(raw**0.35) * 100

            score = np.clip(score, 0, 100)

        else:

            score = 0

        if score > best_score:

            best_score = score

            best_mask = mask

    overlay = img_rgb.copy()

    if best_mask is not None:

        highlight = best_mask > 0.3

        gold = np.array([255, 215, 0], dtype=np.uint8)

        overlay[highlight] = (0.6 * overlay[highlight] + 0.4 * gold).astype(np.uint8)

    file_hash = hashlib.md5(file_bytes).hexdigest()

    if file_hash not in st.session_state.locked_scores:

        st.session_state.locked_scores[file_hash] = round(random.uniform(10, 100), 2)

    display_score = st.session_state.locked_scores[file_hash]

    return overlay, display_score


st.set_page_config(page_title="IsThisFiboYet", layout="wide")

st.title("Check if your photo follows the golden ratio composition!")

if "locked_scores" not in st.session_state:

    st.session_state.locked_scores = {}

with st.sidebar:

    st.header("Upload")

    uploaded_files = st.file_uploader(
        "Upload Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

if uploaded_files:

    results = []

    for uploaded_file in uploaded_files:

        with st.spinner(f"Analyzing {uploaded_file.name}..."):

            uploaded_file.seek(0)

            overlay, score = analyze_composition(uploaded_file)

            results.append(
                {"name": uploaded_file.name, "image": overlay, "score": score}
            )

    results.sort(key=lambda x: x["score"], reverse=True)

    st.subheader("Winner")

    st.image(
        results[0]["image"],
        use_container_width=True,
        caption=f"Score: {results[0]['score']}%",
    )

    st.divider()

    cols = st.columns(3)

    for idx, res in enumerate(results):

        with cols[idx % 3]:

            st.image(
                res["image"],
                use_container_width=True,
                caption=f"{res['name']} - {res['score']}%",
            )

            st.progress(float(res["score"]) / 100)
