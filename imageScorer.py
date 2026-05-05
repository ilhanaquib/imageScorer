import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


def generate_fibonacci_mask(height, width, orientation):
    """
    Generates a deep Golden Ratio mask by recursively subdividing the image.
    Continues until the squares reach a 1-pixel limit for a 'complete' spiral.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Starting boundaries of the "Working Rectangle"
    x, y, w, h = 0, 0, width, height
    squares = []

    # Use a while loop to keep going until we hit the pixel limit
    i = 0
    while True:
        side = min(w, h)
        # Stop only when it's physically impossible to draw a smaller square
        if side < 1:
            break

        current_rect = [int(x), int(y), int(side), int(side)]

        # Subdivision logic: Cut square from the current 'small' side
        if i % 4 == 0:  # Square on Left
            x += side
            w -= side
        elif i % 4 == 1:  # Square on Top
            y += side
            h -= side
        elif i % 4 == 2:  # Square on Right
            current_rect[0] = int(x + w - side)
            w -= side
        elif i % 4 == 3:  # Square on Bottom
            current_rect[1] = int(y + h - side)
            h -= side

        squares.append({"rect": current_rect, "mode": i % 4})
        i += 1

        # Safety break to prevent infinite loops (though side < 1 handles this)
        if i > 50:
            break

    # Draw the calculated geometry
    for s in squares:
        sx, sy, sw, sh = s["rect"]

        # Adaptive thickness ensures the center doesn't become a solid gold blob
        thickness = 2 if sw > 20 else 1

        # 1. Draw the Bounding Box
        cv2.rectangle(mask, (sx, sy), (sx + sw, sy + sh), 255, 1)

        # 2. Draw the Connecting Arc
        if s["mode"] == 0:
            center, angles = (sx + sw, sy + sh), (180, 270)
        elif s["mode"] == 1:
            center, angles = (sx, sy + sh), (270, 360)
        elif s["mode"] == 2:
            center, angles = (sx, sy), (0, 90)
        elif s["mode"] == 3:
            center, angles = (sx + sw, sy), (90, 180)

        cv2.ellipse(mask, center, (sw, sh), 0, angles[0], angles[1], 255, thickness)

    # Orientation flips
    if orientation == 1:
        mask = cv2.flip(mask, 1)
    elif orientation == 2:
        mask = cv2.flip(cv2.flip(mask, 1), 0)
    elif orientation == 3:
        mask = cv2.flip(mask, 0)

    return mask


def analyze_composition(uploaded_file):
    """
    Analyzes the image with a fixed sensitivity of 80.
    """
    FIXED_SENSITIVITY = 80

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, FIXED_SENSITIVITY, FIXED_SENSITIVITY * 2)
    h, w = gray.shape

    best_score = 0
    best_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(4):
        mask = generate_fibonacci_mask(h, w, i)
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        overlap = cv2.bitwise_and(edges, dilated_mask)

        actual_hits = np.count_nonzero(overlap)
        potential_hits = np.count_nonzero(mask)
        score = (actual_hits / potential_hits) * 100 if potential_hits > 0 else 0

        if score > best_score:
            best_score = score
            best_mask = mask

    overlay_img = img_rgb.copy()
    overlay_img[best_mask > 0] = [255, 215, 0]
    return overlay_img, min(round(best_score, 2), 100.0)


# --- Streamlit UI ---
st.set_page_config(page_title="Infinite Fibonacci", layout="wide")
st.title("🌀 Complete Golden Ratio Analyzer")

with st.sidebar:
    st.header("Upload")
    uploaded_files = st.file_uploader(
        "Upload Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    st.info("Sensitivity is locked at 80 for consistent ranking results.")

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        with st.spinner(f"Deep analyzing {uploaded_file.name}..."):
            uploaded_file.seek(0)
            overlay, score = analyze_composition(uploaded_file)
            results.append(
                {"name": uploaded_file.name, "image": overlay, "score": score}
            )

    results.sort(key=lambda x: x["score"], reverse=True)

    st.subheader("🏆 Winner")
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
            st.progress(res["score"] / 100)
