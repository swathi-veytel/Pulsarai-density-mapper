import streamlit as st
import pandas as pd
import io
import os
import cv2 as cv
from google.cloud import storage
import numpy as np
import time

# === Shared Constants ===
GCS_BUCKET_NAME = "veytel-cloud-store"
GCS_FOLDER_PATH = "density_mapper"
SERVICE_ACCOUNT_FILE = st.secrets["gcs_service_account"]
USERS = ["GK", "Siddique", "Nameer", "Taaha", "Konstantine", "Vijayakumar", "Swathi", "Ellen", "Cathy", "Robin", "Anrey", "Clara", "Song", "Kevin",
          "Claire", "Rachel", "Mike", "Paul", "Test_1", "Test_2", "Test_3", "Test_4", "Test_5", "Test_6", "Test_7", "Test_8", "Test_9", "Test_10",
          "Expert_Annotator_1", "Expert_Annotator_2", "Expert_Annotator_3", "Expert_Annotator_4", "Expert_Annotator_5", "Expert_Annotator_6"]
UPMC_USERS = ["GK", "Siddique", "Nameer", "Taaha", "Konstantine"]

color_map = {
    0: (129, 212, 250),  # blue
    1: (152, 255, 152),  # mint green
    2: (229, 185, 0),    # Orange
    3: (255, 0, 0)       # red
}

@st.cache_resource
def authenticate_gcs():
    return storage.Client.from_service_account_info(SERVICE_ACCOUNT_FILE)

def download_csv(user):
    client = authenticate_gcs()
    bucket = client.get_bucket(GCS_BUCKET_NAME)
    filename = f"density_mapper_v8_{user}.csv"
    blob = bucket.blob(os.path.join(GCS_FOLDER_PATH, filename))
    if blob.exists():
        csv_content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_content))
        df["user"] = user
        return df
    return pd.DataFrame()

def load_image(image_path):
    if os.path.exists(image_path):
        return cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    return np.zeros((256, 256), dtype=np.uint8)

def apply_thresholds_overlay(textured_cxr, lung_mask, t0, t1, t2):
    lung_mask_bool = lung_mask > 0 if lung_mask is not None else np.ones_like(textured_cxr, dtype=bool)

    dense_0 = np.where((textured_cxr < t0) & lung_mask_bool, textured_cxr, 0)
    dense_1 = np.where((textured_cxr >= t0) & (textured_cxr < t1) & lung_mask_bool, textured_cxr, 0)
    dense_2 = np.where((textured_cxr >= t1) & (textured_cxr < t2) & lung_mask_bool, textured_cxr, 0)
    dense_3 = np.where((textured_cxr >= t2) & lung_mask_bool, textured_cxr, 0)

    return [dense_0, dense_1, dense_2, dense_3]

def apply_density_diff(cxr, textured_cxr, mask):
    diff = textured_cxr.astype(np.float32) - cxr.astype(np.float32)
    masked_diff = np.where(mask > 0, diff, 0)
    return np.clip(cxr + masked_diff, 0, 255).astype(np.uint8)

def overlay_dense_pixels(base_img, densities, up_to=3, alpha=0.3):
    base_color = cv.cvtColor(base_img, cv.COLOR_GRAY2BGR)
    blended = base_color.copy()
    for i in range(up_to + 1):
        mask = densities[i] > 0
        mask_3ch = np.stack([mask]*3, axis=-1)
        overlay = np.zeros_like(base_color, dtype=np.uint8)
        overlay[:, :] = color_map[i]
        blended[mask_3ch] = ((1 - alpha) * blended[mask_3ch] + alpha * overlay[mask_3ch]).astype(np.uint8)
    return blended

def show_results():
    st.title("\U0001F4CA Annotator Results Viewer")

    view_option = st.radio("Select Results View Mode:",
                           ["View results from UPMC Users", "Select Annotators to Compare", "View results All users"],
                           index=0)

    overlay_mode = st.checkbox("Show Colored Overlays", value=True)

    if "zoom_slider_val" not in st.session_state:
        st.session_state.zoom_slider_val = 1
    if "prev_zoom_slider" not in st.session_state:
        st.session_state.prev_zoom_slider = 1
    if "zoom_level" not in st.session_state:
        st.session_state.zoom_level = 200

    zoom_col1, _, _, _ = st.columns(4)
    with zoom_col1:
        zoom_slider = st.slider("Zoom Level", 1, 10, value=st.session_state.zoom_slider_val, key="zoom_slider")

    if zoom_slider != st.session_state.prev_zoom_slider:
        st.session_state.zoom_slider_val = zoom_slider
        st.session_state.zoom_level = 200 + (zoom_slider - 1) * 30
        st.session_state.prev_zoom_slider = zoom_slider
        with st.spinner("Updating zoom level..."):
            time.sleep(0.5)
        st.rerun()

    image_display_width = st.session_state.zoom_level

    if view_option == "Select Annotators to Compare":
        selected_users = st.multiselect("Choose Annotators to View", USERS)
        if not selected_users:
            st.warning("Please select users to compare.")
            return
    elif view_option == "View results from UPMC Users":
        selected_users = UPMC_USERS
    else:
        selected_users = USERS

    with st.spinner("Loading image data..."):
        all_data = pd.concat([download_csv(user) for user in selected_users if not download_csv(user).empty], ignore_index=True)

    if all_data.empty:
        st.info("No annotation data found for selected users.")
        return

    images = all_data["imagName"].drop_duplicates().tolist()

    image_path_base = "Images"
    base_cxr_path = os.path.join(image_path_base, "cxr")
    synth_cxr_path = os.path.join(image_path_base, "textured_cxr")
    lung_mask_path = os.path.join(image_path_base, "mask")

    with st.spinner("Rendering image previews..."):
        for image_name in images[:10]:
            st.markdown(f"## \U0001F4F7 Image: `{image_name}`")
            cxr = load_image(os.path.join(base_cxr_path, f"cxr_{image_name}"))
            synthetic = load_image(os.path.join(synth_cxr_path, f"synthetic_{image_name}"))
            lung_mask = load_image(os.path.join(lung_mask_path, f"mask_{image_name}"))

            col1, col2 = st.columns(2)
            with col1:
                st.image(cxr, caption="Original CXR", width=image_display_width)
            with col2:
                st.image(synthetic, caption="Synthetic CXR", width=image_display_width)

            for user in selected_users:
                user_df = all_data[(all_data.user == user) & (all_data.imagName == image_name)]
                if user_df.empty:
                    continue

                st.markdown(f"### \U0001F464 {user}")
                row = user_df.iloc[0]
                t0, t1, t2 = int(row["max_thresh0"]), int(row["max_thresh1"]), int(row["max_thresh2"])
                max_density = int(row["max_density"])
                st.markdown(f"**Max Density Marked:** Density {max_density}")

                densities = apply_thresholds_overlay(synthetic, lung_mask, t0, t1, t2)

                col_dens = st.columns(4)
                for d in range(max_density + 1):
                    caption = f"Density {d}"
                    if d == 0:
                        caption += f"\n(pixels < {t0})"
                    elif d == 1:
                        caption += f"\n({t0}-{t1})"
                    elif d == 2:
                        caption += f"\n({t1}-{t2})"
                    else:
                        caption += f"\n(>= {t2})"

                    if overlay_mode:
                        vis_img = overlay_dense_pixels(cxr, densities, up_to=d)
                    else:
                        progressive_mask = sum(densities[:d+1])
                        vis_img = apply_density_diff(cxr, synthetic, progressive_mask)

                    with col_dens[d]:
                        st.image(vis_img, caption=caption, width=image_display_width)
            st.divider()
