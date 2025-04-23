import streamlit as st
import os
import cv2 as cv
import numpy as np
from google.cloud import storage
import pandas as pd
import time


# Google Cloud configuration
GCS_BUCKET_NAME = "veytel-cloud-store"
GCS_FOLDER_PATH = "density_mapper"

#gets service account key from secrets
SERVICE_ACCOUNT_FILE = st.secrets["gcs_service_account"]

USERS = ["GK", "Siddique", "Nameer", "Taaha", "Konstantine","Vijayakumar","Swathi", "Ellen", "Cathy", "Robin", "Anrey", "Clara",  "Song", "Kevin",
                                            "Claire", "Rachel", "Mike", "Paul", "Test_1", "Test_2", "Test_3", "Test_4", "Test_5", "Test_6", "Test_7", "Test_8", "Test_9", "Test_10", "Expert_Annotator_1", "Expert_Annotator_2", "Expert_Annotator_3", "Expert_Annotator_4", "Expert_Annotator_5", "Expert_Annotator_6"]

prefix = ""

st.set_page_config(layout="wide")
st.markdown("""
        <style>
        h1 { font-size: 16px !important; }
       /* Reduce width of the main container */
        .main .block-container {
            max-width: 100%; /* Adjust this value */
        }
            /* Reduce top margin */
            .block-container {
                padding-top:10px !important;
            }
        /* Reduce expander size */
        div.streamlit-expanderContent {
            font-size: 14px !important;
            padding: 2px !important;
        }
        div.streamlit-expanderHeader {
            font-size: 10px !important;
            width: 200px !important;
            height: 50px
        }
        </style>
    """, unsafe_allow_html=True)



st.markdown("""
            <style>
            
                div.stButton > button {
                    padding: 4px 6px;  /* Reduce top/bottom padding */
                    font-size: 11px;    /* Optional: Smaller font */
                }
        
            /* Reduce height and font size for text input */
            input[type="text"] {
                padding: 4px 6px;
                font-size: 12px;
            }
            div[role="radiogroup"] label {
        font-size: 11px !important;
        line-height: 1.2 !important;
    }

            /* Reduce font size of slider labels */
            .stSlider > div[data-baseweb="slider"] {
                padding-top: 0.1rem;
                padding-bottom: 0.1rem;
            }
            .stSlider label, .stSlider span {
                font-size: 11px;
            }
            </style>
        """, unsafe_allow_html=True)
color_map = {
            0: (129, 212, 250), # blue
            1: (152, 255, 152), # mint green
            2: (229, 185, 0),  # Orange
            3: (255, 0, 0)  # red
        }

#2: (255, 105, 180),  # Pink
#0: (50, 205, 50),  # Lime Green

csv_path = "Images/density_mapper_data_Mar_25.csv"  # update this to the actual path
data_df = pd.read_csv(csv_path)

# Function to authenticate GCS client
def authenticate_gcs():
    return storage.Client.from_service_account_info(SERVICE_ACCOUNT_FILE)

# Function to download CSV from GCS
def download_csv_from_gcs(filename):
    client = authenticate_gcs()
    bucket = client.get_bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(os.path.join(GCS_FOLDER_PATH, filename))
    csv_content = blob.download_as_text()
    return csv_content

# Function to read user-specific CSV data
def read_csv_from_gcs(user):
    filename = f"density_mapper_v8_{user}.csv"
    try:
        csv_content = download_csv_from_gcs(filename)
        rows = csv_content.strip().split("\n")
        csv_data = [row.split(",") for row in rows]
        return csv_data
    except Exception as e:
        return []

# Function to upload CSV to GCS
def upload_csv_to_gcs(csv_content, filename):
    client = authenticate_gcs()
    bucket = client.get_bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(os.path.join(GCS_FOLDER_PATH, filename))
    blob.upload_from_string(csv_content)

# Function to create or append to CSV for the user
def create_csv(user, imagName, count, max_thresh0, max_thresh1, max_thresh2, max_density):
    csv_data = read_csv_from_gcs(user)
    if not csv_data:
        csv_content = "imagName,count,max_thresh0,max_thresh1,max_thresh2,max_density"
    else:
        csv_content = "\n".join([",".join(row) for row in csv_data])
    csv_content += f"\n{imagName},{count},{max_thresh0},{max_thresh1},{max_thresh2},{max_density}\n"
    filename = f"density_mapper_v8_{user}.csv"
    upload_csv_to_gcs(csv_content, filename)

# Function to apply thresholds and get different density masks
def apply_thresholds_overlay(cxr, textured_cxr, lung_mask, thresh1, thresh2, thresh3):
    #max_val = np.percentile(cxr, 97)
    #scaled_thresh1 = thresh1 * max_val / 255
    #scaled_thresh2 = thresh2 * max_val / 255
    #scaled_thresh3 = thresh3 * max_val / 255

    if lung_mask is None:
        st.error("Lung mask image could not be loaded.")
        return

    # Ensure lung_mask is boolean
    lung_mask_bool = lung_mask > 0

    # Apply thresholds within lung mask
    st.session_state.dense_0 = np.where(
        (textured_cxr < thresh1) & lung_mask_bool, textured_cxr, 0
    )

    st.session_state.dense_1 = np.where(
        (textured_cxr >= 0) & (textured_cxr < thresh2) & lung_mask_bool,
        textured_cxr,
        0,
    )

    st.session_state.dense_2 = np.where((textured_cxr >= 0) & (textured_cxr < thresh3) & lung_mask_bool, textured_cxr,
        0,
    )

    st.session_state.dense_3 = np.where(
        (textured_cxr >= 0) & lung_mask_bool, textured_cxr, 0
    )


def apply_thresholds_from_synthetic_range(cxr, textured_cxr, thresh1, thresh2, thresh3):
    # Convert to float to avoid underflow
    cxr_f = cxr.astype(np.float32)
    textured_f = textured_cxr.astype(np.float32)

    # Compute difference image
    diff = textured_f - cxr_f

    # Create masks based on pixel intensity of synthetic image
    mask_0 = (textured_f < thresh1)
    mask_1 = (textured_f >= thresh1) & (textured_f < thresh2)
    mask_2 = (textured_f >= thresh2) & (textured_f < thresh3)
    mask_3 = (textured_f >= thresh3)

    # Apply each mask to the diff (preserving sign)
    dense_0 = np.where(mask_0, diff, 0)
    dense_1 = np.where(mask_1, diff, 0)
    dense_2 = np.where(mask_2, diff, 0)
    dense_3 = np.where(mask_3, diff, 0)

    # Store in session_state as float for precision, clip only for display
    st.session_state.dense_0 = dense_0
    st.session_state.dense_1 = dense_1
    st.session_state.dense_2 = dense_2
    st.session_state.dense_3 = dense_3


def overlay_dense_pixels(base_img, color=0, alpha=0.2):
    """
    Overlay exclusive density masks onto base_img.
    Each pixel is colored according to which density range it falls into.
    Densities are mutually exclusive.

    color: int
        0 - Only dense_0 (Pink)
        1 - dense_0 (Pink), dense_1 (Cyan)
        2 - + dense_2 (Lime Green)
        3 - + dense_3 (Orange)
    """

    # Define color map for overlays

    # Convert base grayscale image to 3-channel BGR
    base_color = cv.cvtColor(base_img, cv.COLOR_GRAY2BGR)
    blended = base_color.copy()

    # Go through each density level up to the selected one
    for i in range(color + 1):
        dense_mask = st.session_state.get(f"dense_{i}", None)
        if dense_mask is None:
            continue

        # Create binary mask for this level only (non-zero values)
        binary_mask = (dense_mask > 0).astype(np.uint8)
        mask_3ch = np.stack([binary_mask] * 3, axis=-1)

        # Prepare color overlay
        overlay = np.zeros_like(base_color, dtype=np.uint8)
        overlay[:, :] = color_map[i]

        # Blend the selected color onto the base image where this density level is active
        blended[mask_3ch == 1] = (
            (1 - alpha) * blended[mask_3ch == 1] + alpha * overlay[mask_3ch == 1]
        ).astype(np.uint8)

        # Once a pixel is assigned to a density level, it should not be considered again
        # So we zero it out from the remaining dense masks
        for j in range(i + 1, 4):
            next_mask = st.session_state.get(f"dense_{j}", None)
            if next_mask is not None:
                st.session_state[f"dense_{j}"] = np.where(binary_mask == 1, 0, next_mask)

    return blended

def apply_density_diff(cxr, textured_cxr, density_mask):
    """
    Reconstruct a CXR image by adding only the pixel-wise difference
    from textured_cxr to cxr at positions defined by the density mask.

    Parameters:
    - cxr: np.ndarray, original grayscale CXR image (float32 or uint8)
    - textured_cxr: np.ndarray, noised/textured version of CXR (same shape)
    - density_mask: np.ndarray, binary mask or thresholded map to select regions to add diff

    Returns:
    - added_image: np.ndarray, reconstructed image with partial difference added
    """

    # Ensure float32 for arithmetic
    cxr = cxr.astype(np.float32)
    textured_cxr = textured_cxr.astype(np.float32)

    # Compute the difference
    diff = textured_cxr - cxr

    # Get diff only at selected density mask locations
    selected_diff = np.where(density_mask > 0, diff, 0)

    # Add to original CXR and clip to valid pixel range
    added_image = np.clip(cxr + selected_diff, 0, 255).astype(np.uint8)

    return added_image

def apply_gamma_correction(image, gamma=1.2):
    # Normalize image to [0, 1] range if needed
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    # Apply gamma correction
    corrected = np.power(image, 1.0 / gamma)

    # Convert back to 8-bit
    return (corrected * 255).clip(0, 255).astype(np.uint8)



# Image loading function
def load_images(image_id, index, prefix=""):
    if st.session_state.image_id >= len(data_df):
        st.info("Processing complete! No further images to process.")
        st.stop()

    #print(len(data_df))
    row = data_df.iloc[image_id]

    # Correct mapping based on actual folder and filename types
    cxr_path = os.path.join(prefix, "Images/cxr", row["cxr"])
    lung_noised_path = os.path.join(prefix, "Images/textured_cxr", row["synthetic"])
    synthetic_path = os.path.join(prefix, "Images/lung_noised", row["noise"])
    mask_path = os.path.join(prefix, "Images/mask", row["mask"])
    #synthetic_path = os.path.join(prefix, "Images/textured_cxr", row["synthetic"])

    print("index:", index)
    print("CXR:", cxr_path)
    print("Lung Noised:", lung_noised_path)
    print("Synthetic:", synthetic_path)

    # Read images
    cxr = cv.imread(cxr_path,  cv.IMREAD_GRAYSCALE)
    #cxr_gamma_corrected = apply_gamma_correction(cxr)
    #cv.imread(image_path, cv.IMREAD_UNCHANGED)
    lung_noised = cv.imread(lung_noised_path,  cv.IMREAD_GRAYSCALE)
    textured_cxr = cv.imread(synthetic_path,  cv.IMREAD_GRAYSCALE)
    lung_mask = cv.imread(mask_path,  cv.IMREAD_GRAYSCALE)
    return cxr, textured_cxr, lung_noised, lung_mask

user_password = st.secrets["general"]["user_password"]
# Authentication logic
def check_auth(username, password):
    if username in USERS:
        if password == user_password:
            return username
    return None


# App main function
def main():
    #st.set_page_config(layout="centered")
    # Session state initialization
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'show_brightness' not in st.session_state:
        st.session_state.show_brightness = False
    if 'count' not in st.session_state:
        st.session_state.count = 0
    if 'image_id' not in st.session_state:
        st.session_state.image_id = 1
    if 'index' not in st.session_state:
        st.session_state.index = 1
    if 'fresh_start' not in st.session_state:
        st.session_state.fresh_start = True
    # Initialize threshold values in session state if not already done
    if 'max_thresh0' not in st.session_state:
        st.session_state.max_thresh0 = 50  # Default value

    if 'max_thresh1' not in st.session_state:
        st.session_state.max_thresh1 = 100  # Default value

    if 'max_thresh2' not in st.session_state:
        st.session_state.max_thresh2 = 150  # Default value
    if 'dense_0' not in st.session_state:
        st.session_state.dense_0 = 50  # Default value
        # Initialize threshold values in session state if not already done
    if 'dense_1' not in st.session_state:
        st.session_state.dense_1 = 100  # Default value

    if 'dense_2' not in st.session_state:
        st.session_state.dense_2 = 150  # Default value

    if 'dense_3' not in st.session_state:
        st.session_state.dense_3 = 200  # Default value

    if "overlay_toggle" not in st.session_state:
        st.session_state.overlay_toggle = True  # default to checked

    if "user_changed_density" not in st.session_state:
        st.session_state.user_changed_density = False

    if "max_density_selection" not in st.session_state:
        st.session_state.max_density_selection = "Density 3"
        st.session_state.selected_max_density = 3
    if "show_instructions" not in st.session_state:
        st.session_state.show_instructions = False

    def get_image_id_index(count):
        # global image_id, index
        #todo: modify for simgle image
        if count == 0:
            return 0, 1
        #image_id = (count - 1) // 3 + 1
        #index = (count - 1) % 3 + 1
        image_id = count  # Directly map count to image_id
        index = 1 # No need for different indices, since each image is unique
        print("getting count, image_id, index", count, image_id, index)
        return image_id, index

    # Check if count exceeds the limit (e.g., 30)
    if st.session_state.image_id >= len(data_df):
        st.info("No more images available. You have reached the end of the image set.")
        st.stop()

    #dense_0, dense_1, dense_2, dense_3

    # Load images based on image_id and index
    #cxr, textured_cxr, lung_noised = load_images(st.session_state.image_id, st.session_state.index)

    # Login logic
    if st.session_state.user is None and not st.session_state.show_brightness:
        st.title("Login")
        username = st.selectbox("Select User", USERS)
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = check_auth(username, password)
            if user:
                st.session_state.user = user
                st.success(f"Welcome {user}!")
                st.session_state.show_brightness = True

                st.rerun()
            else:
                st.error("Invalid credentials")
    elif st.session_state.show_brightness:
        st.success(f"Welcome {st.session_state.user}!")
        st.spinner("Please increase your screen brightness...")
        st.markdown("### 🔆 Tip: Increase your screen brightness for better visibility.")
        if st.button("Proceed"):
            st.session_state.show_brightness = False
            st.rerun()
        st.image("brightness.jpg", caption="How to increase brightness on Mac")


    else:
        st.session_state.show_brightness = False
        st.title("Density Mapper")

        csv_data = read_csv_from_gcs(st.session_state.user)
        st.session_state.count = int(csv_data[-1][1])+1 if csv_data and len(csv_data) > 1 else 0
        print("count from csv :",st.session_state.count)
        st.session_state.image_id, st.session_state.index = get_image_id_index(st.session_state.count)
        print("starting count, image id, idx", st.session_state.count, st.session_state.image_id, st.session_state.index)

        if (st.session_state.count == 100):
            empty_image = np.zeros((256, 256), dtype=np.uint8)
            cxr = empty_image
            lung_mask = empty_image
            textured_cxr = empty_image
            lung_noised = empty_image

        else:
            # Load images based on image_id and index
            cxr, textured_cxr, lung_noised, lung_mask = load_images(st.session_state.image_id, st.session_state.index)

        # Top row: Original CXR, Noise, Synthetic CXR
        col1, col2, col3, col4 = st.columns(4)
        if st.session_state.count < len(data_df):
            image_name = data_df.iloc[st.session_state.count]['imgName']
        else:
            image_name = "Empty"
        image_name = image_name.replace(".png", "")
        with col1:
            st.image(cxr, caption="Original CXR", width=200)
        #with col2:
        #    st.image(textured_cxr, caption="Noise", width=200)
        with col2:
            st.image(lung_noised, caption="Synthetic CXR", width=200)
        with col3:
            # Save button and progress tracking
            progress = st.slider("Progress", 0, 100, value=st.session_state.count, key="progress_slider", disabled=True)

            label = "Save & Continue"
            if(st.session_state.count == 0):
                label = "Start"
                # Expandable instructions section

            if st.button(label):
                if st.session_state.count >= len(data_df):
                    st.info("Processing complete 🎉! No further images to process. You may close the window!")
                    return

                print("saving count, image id, idx", st.session_state.count, st.session_state.image_id,
                      st.session_state.index)
                create_csv(st.session_state.user, data_df.iloc[st.session_state.count]['imgName'], st.session_state.count, st.session_state.max_thresh0,
                           st.session_state.max_thresh1, st.session_state.max_thresh2, st.session_state.selected_max_density)
                st.success(f"Data saved!")

                st.session_state.count += 1
                st.session_state.image_id, st.session_state.index = get_image_id_index(st.session_state.count)

                st.session_state["max_density_selection"] = "Density 3"
                st.session_state.selected_max_density = 3
                st.session_state.user_changed_density = False
                st.rerun()


            #instructions:

            if st.session_state.show_instructions:
                if st.button("❌ Hide Instructions"):
                    st.session_state.show_instructions = False
                    st.rerun()
            else:
                if st.button("💡 Show Instructions"):
                    st.session_state.show_instructions = True
                    st.rerun()

            # Scrollable floating instruction box
            if st.session_state.show_instructions:
                st.markdown("""
                    <style>
                    /* Force scrollbar visibility inside the floating box */
                    .instruction-box::-webkit-scrollbar {
                        width: 6px;
                    }

                    .instruction-box::-webkit-scrollbar-thumb {
                        background-color: #ccc;
                        border-radius: 4px;
                    }

                    .instruction-box::-webkit-scrollbar-track {
                        background-color: #f1f1f1;
                    }
                    </style>

                    <div class="instruction-box" style="
                        position: fixed;
                        top: 80px;
                        right: 20px;
                        width: 320px;
                        max-height: 300px;
                        overflow-y: scroll;
                        background-color: #f9f9f9;
                        padding: 15px;
                        border: 1px solid #ccc;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        z-index: 1000;
                        font-size: 13px;
                    ">
                    <b>📝 Instructions (Scroll Down for more!)</b>
                    <ul>
                        <li>🔆 Set your screen brightness to maximum for best visibility.</li>
                        <li>The synthetic image (right) is created by adding synthetic noise to the original CXR (left).</li>
                        <li><strong>Max Density:</strong> If you think the synthetic CXR only goes up to, say, Density 2, select Density 2 as Maximum Density.</li>
                        <li><strong>Overlay mode ON:</strong> Highlights selected pixel ranges on the synthetic image using unique colors for each density level.</li>
                        <li><strong>Overlay mode OFF:</strong> Adds the selected pixel ranges directly to the original image without any color highlights.</li>
                        <li>Use the sliders to adjust pixel thresholds and define them at each density level.</li>
                        <li><strong>Progressive addition:</strong> At Density 2, you’ll see all pixels from Density 0 to 2 combined and so on.</li>
                        <li>Click “Save & Continue” to move to the next image. Your progress is tracked.</li>
                        <li>You can close the window anytime. Your place will be saved automatically.</li></ul>
                    </div>
                """, unsafe_allow_html=True)
                # Re-render the progress bar with the updated count
                #st.slider("Progress", 1, 30, value=st.session_state.count, key="progress_slider", disabled=True)
            st.markdown(f"<div style='text-align: center; font-size: 0.9em; color: gray;'>Image: {image_name}</div>", unsafe_allow_html=True)

            with col4:
                if st.session_state.overlay_toggle:
                    scale_image_path = os.path.join("Images", "density_scale.png")
                    scale_image = cv.imread(scale_image_path)

                    if scale_image is not None:
                        scale_image = cv.cvtColor(scale_image, cv.COLOR_BGR2RGB)  # Convert to RGB
                        st.image(scale_image, caption="color scale", width=150)
                    else:
                        st.warning("Could not load the scale image.")


        apply_thresholds_overlay(cxr, lung_noised, lung_mask, st.session_state.max_thresh0,
                         st.session_state.max_thresh1,
                         st.session_state.max_thresh2)


        # Horizontal line to separate rows
        #st.divider()
        if st.session_state.fresh_start:
            st.session_state.max_density_selection = "Density 3"
            st.session_state.selected_max_density = 3
            st.session_state.fresh_start = False  # prevent resetting again

        # Top row: Original CXR, Noise, Synthetic CXR
        but_col1, but_col2, but_col3 = st.columns(3)
        with but_col1:
            # Render the checkbox with a temp key
            overlay_temp = st.checkbox("**Colored Overlay Mode On**", value=st.session_state.overlay_toggle, key="overlay_temp")

            # Only update session state if it changed
            if overlay_temp != st.session_state.overlay_toggle:
                st.session_state.overlay_toggle = overlay_temp
                st.rerun()

        with but_col2:
            prev_value = st.session_state.get("selected_max_density", None)
            selected_max_density = st.radio(
                "**Select Maximum Density in Synthetic CXR:**",
                options=["Density 0", "Density 1", "Density 2", "Density 3"],
                key="max_density_selection",
                horizontal=True
            )

            new_value = int(selected_max_density[-1])
            if prev_value is not None and new_value != prev_value:
                st.session_state.user_changed_density = True

            st.session_state.selected_max_density = new_value

            #st.rerun()
        #else:
            #st.rerun()

        # Bottom row: Density maps with one max slider for each
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.session_state.overlay_toggle:
                highlighted_dense_0 = overlay_dense_pixels(lung_noised, 0)
                st.image(highlighted_dense_0, caption="CXR + Density 0", width=200)
            else:
                #cxr_with_dense_0 = cv.add(cxr, st.session_state.dense_0.astype(np.uint8))
                diff = textured_cxr - cxr
                #todo: get pixels from dense 0 from diff , when adding instead of adding dense_0 add pixels from diff at dnse 0

                #cxr_with_dense_0 = np.clip(cxr.astype(np.float32) + st.session_state.dense_0, 0, 255).astype(np.uint8)
                cxr_with_dense_0 = apply_density_diff(cxr, lung_noised, st.session_state.dense_0)
                #cxr_with_dense_0 = cxr + st.session_state.dense_0.astype(np.uint8)
                st.image(cxr_with_dense_0, caption="CXR + Density 0", width=200)


            min_density0 = 0
            st.text_input("Min Density 0", value=min_density0, disabled=True)



            # Allow slider to go up to 256, but display max as 255
            max_slider0 = st.slider("Max Density 0", min_density0, 255,
                                    value=st.session_state.max_thresh0, key="max_slider0")

            if max_slider0 != st.session_state.max_thresh0:
                st.session_state.max_thresh0 = max_slider0
                if st.session_state.overlay_toggle:
                    apply_thresholds_overlay(cxr, lung_noised, lung_mask, st.session_state.max_thresh0,
                                 st.session_state.max_thresh1, st.session_state.max_thresh2)
                else:
                    apply_thresholds_from_synthetic_range(cxr, lung_noised, st.session_state.max_thresh0,
                                 st.session_state.max_thresh1, st.session_state.max_thresh2)
                st.rerun()

            # If Density 0 is set to 255, force the rest to 255
            max_slider1_disabled = max_slider0 >= 255
            max_slider2_disabled = max_slider1_disabled or st.session_state.max_thresh1 >= 255

        with col2:

            if st.session_state.overlay_toggle:
                highlighted_dense_1 = overlay_dense_pixels(lung_noised, 1)
                #cxr_with_dense_1 = cv.add(cxr, st.session_state.dense_1.astype(np.uint8))
                st.image(highlighted_dense_1, caption="CXR + Density 1", width=200)
            else:
                #cxr_with_dense_1 = np.clip(cxr.astype(np.float32) + st.session_state.dense_1, 0, 255).astype(np.uint8)
                cxr_with_dense_1 = apply_density_diff(cxr, lung_noised, st.session_state.dense_1)
                st.image(cxr_with_dense_1, caption="CXR + Density 1", width=200)


            min_density1 = max_slider0
            st.text_input("Min Density 1", value=min_density1, disabled=True)

            max_slider1_value = 255 if max_slider1_disabled else max(st.session_state.max_thresh1, min_density1)
            max_slider1 = st.slider("Max Density 1", min_density1, 256,
                                    value=max_slider1_value, key="max_slider1", disabled=max_slider1_disabled)

            if not max_slider1_disabled and max_slider1 != st.session_state.max_thresh1:
                st.session_state.max_thresh1 = max_slider1
                if st.session_state.overlay_toggle:
                    apply_thresholds_overlay(cxr, lung_noised, lung_mask, st.session_state.max_thresh0,
                                 st.session_state.max_thresh1, st.session_state.max_thresh2)
                else:
                    apply_thresholds_from_synthetic_range(cxr, lung_noised, st.session_state.max_thresh0,
                                 st.session_state.max_thresh1, st.session_state.max_thresh2)
                st.rerun()

            # Disable next slider if max_slider1 is 255
            max_slider2_disabled = max_slider1 >= 255

        with col3:
            if st.session_state.overlay_toggle:
                highlighted_dense_2 = overlay_dense_pixels(lung_noised, 2)
                #cxr_with_dense_2 = cv.add(cxr, st.session_state.dense_2.astype(np.uint8))
                st.image(highlighted_dense_2, caption="CXR + Density 2", width=200)
            else:
                #cxr_with_dense_2 = cv.add(cxr, st.session_state.dense_2.astype(np.uint8))
                cxr_with_dense_2 = apply_density_diff(cxr, lung_noised, st.session_state.dense_2)
                st.image(cxr_with_dense_2, caption="CXR + Density 2", width=200)

            min_density2 = max_slider1
            st.text_input("Min Density 2", value=min_density2, disabled=True)

            max_slider2_value = 255 if max_slider2_disabled else max(st.session_state.max_thresh2, min_density2)
            max_slider2 = st.slider("Max Density 2", min_density2, 256,
                                    value=max_slider2_value, key="max_slider2", disabled=max_slider2_disabled)

            if not max_slider2_disabled and max_slider2 != st.session_state.max_thresh2:
                st.session_state.max_thresh2 = max_slider2
                if st.session_state.overlay_toggle:
                    apply_thresholds_overlay(cxr, lung_noised, lung_mask, st.session_state.max_thresh0,
                                 st.session_state.max_thresh1, st.session_state.max_thresh2)
                else:
                    apply_thresholds_from_synthetic_range(cxr, lung_noised, st.session_state.max_thresh0,
                                             st.session_state.max_thresh1, st.session_state.max_thresh2)
                st.rerun()

        with col4:
            if st.session_state.overlay_toggle:
                highlighted_dense_3 = overlay_dense_pixels(lung_noised, 3)
                #cxr_with_dense_3 = cv.add(cxr, st.session_state.dense_3.astype(np.uint8))
                st.image(highlighted_dense_3, caption="CXR + Density 3", width=200)
            else:
                #cxr_with_dense_3 = cv.add(cxr, st.session_state.dense_3.astype(np.uint8))
                cxr_with_dense_3 = apply_density_diff(cxr, lung_noised, st.session_state.dense_3)
                st.image(cxr_with_dense_3, caption="CXR + Density 3", width=200)


            st.text_input("Min Density 3", value=st.session_state.max_thresh2 if not max_slider2_disabled else 255,
                          disabled=True)
            st.text_input("Max Density 3", value=255, disabled=True)
            if st.session_state.overlay_toggle:

                apply_thresholds_overlay(cxr, lung_noised, lung_mask, st.session_state.max_thresh0,
                             st.session_state.max_thresh1, st.session_state.max_thresh2)
            else:
                apply_thresholds_from_synthetic_range(cxr, lung_noised, st.session_state.max_thresh0,
                                         st.session_state.max_thresh1, st.session_state.max_thresh2)




        #st.session_state.selected_max_density = int(st.session_state["max_density_selection"][-1])
        #print("Max Density  : ",int(st.session_state["max_density_selection"][-1]))
        #print("Max Density ss  : ", selected_max_density)


# Run the app
if __name__ == "__main__":
    main()
