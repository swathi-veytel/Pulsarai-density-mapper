import streamlit as st
import os
import cv2 as cv
import numpy as np
from google.cloud import storage


# Google Cloud configuration
GCS_BUCKET_NAME = "veytel-cloud-store"
GCS_FOLDER_PATH = "density_mapper"

#gets service account key from secrets
SERVICE_ACCOUNT_FILE = st.secrets["gcs_service_account"]

USERS = ["GK", "Siddique", "Nameer", "Taaha", "Konstantine","Vijayakumar","Swathi", "Ellen", "Cathy", "Robin", "Anrey", "Song", "Kevin",
                                            "Claire", "Rachel", "Mike", "Paul", "Test_1", "Test_2", "Test_3", "Expert_Annotator_1", "Expert_Annotator_2", "Expert_Annotator_3", "Expert_Annotator_4", "Expert_Annotator_5", "Expert_Annotator_6"]

prefix = ""

st.set_page_config(layout="wide")
st.markdown("""
        <style>
        h1 { font-size: 20px !important; }
       /* Reduce width of the main container */
        .main .block-container {
            max-width: 100%; /* Adjust this value */
        }
            /* Reduce top margin */
            .block-container {
                padding-top:20px !important;
            }
            /* Reduce button size */
        div.stButton > button {
            font-size: 10px !important;
            padding: 5px 10px !important;
            width: 200px 
            height: 50 px
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
    filename = f"density_mapper_v3_{user}.csv"
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
def create_csv(user, count, max_thresh0, max_thresh1, max_thresh2):
    csv_data = read_csv_from_gcs(user)
    if not csv_data:
        csv_content = "count,max_thresh0,max_thresh1,max_thresh2"
    else:
        csv_content = "\n".join([",".join(row) for row in csv_data])
    csv_content += f"\n{count},{max_thresh0},{max_thresh1},{max_thresh2}\n"
    filename = f"density_mapper_v3_{user}.csv"
    upload_csv_to_gcs(csv_content, filename)

# Function to apply thresholds and get different density masks
def apply_thresholds(cxr, textured_cxr, thresh1, thresh2, thresh3):
    max_val = np.percentile(cxr, 97)
    scaled_thresh1 = thresh1 * max_val / 255
    scaled_thresh2 = thresh2 * max_val / 255
    scaled_thresh3 = thresh3 * max_val / 255

    st.session_state.dense_0 = np.where(textured_cxr < scaled_thresh1, textured_cxr, 0)
    #st.session_state.dense_1 = np.where(((textured_cxr < scaled_thresh2) & (textured_cxr >= scaled_thresh1)), textured_cxr, 0)
    #st.session_state.dense_2 = np.where(((textured_cxr < scaled_thresh3) & (textured_cxr >= scaled_thresh2)), textured_cxr, 0)
    #st.session_state.dense_3 = np.where(((textured_cxr < 255) & (textured_cxr >= scaled_thresh3)), textured_cxr, 0)

    st.session_state.dense_1 = np.where((textured_cxr < scaled_thresh2), textured_cxr, 0)
    st.session_state.dense_2 = np.where((textured_cxr < scaled_thresh3), textured_cxr, 0)
    st.session_state.dense_3 = np.where((textured_cxr < 255), textured_cxr, 0)


# Image loading function
def load_images(image_id, index):
    cxr_file = f"cxr{image_id}_cxr.png"
    textured_cxr_file = f"cxr{image_id}_textured_{index}.png"
    lung_noised_file = f"cxr{image_id}_lung_noised_{index}.png"

    cxr_path = os.path.join(prefix+"Images/cxr", cxr_file)
    textured_cxr_path = os.path.join(prefix+"Images/textured_cxr", textured_cxr_file)
    lung_noised_path = os.path.join(prefix+"Images/lung_noised", lung_noised_file)

    cxr = cv.imread(cxr_path, cv.IMREAD_GRAYSCALE)
    textured_cxr = cv.imread(textured_cxr_path, cv.IMREAD_GRAYSCALE)
    lung_noised = cv.imread(lung_noised_path, cv.IMREAD_GRAYSCALE)

    return cxr, textured_cxr, lung_noised

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



    def get_image_id_index(count):
        # global image_id, index
        #todo: modify for simgle image
        if count == 0:
            return 1, 1
        #image_id = (count - 1) // 3 + 1
        #index = (count - 1) % 3 + 1
        image_id = count  # Directly map count to image_id
        index = 1 # No need for different indices, since each image is unique
        print("getting count, image_id, index", count, image_id, index)
        return image_id, index

    # Check if count exceeds the limit (e.g., 30)
    if st.session_state.count >= 30:
        st.info("Processing complete! You have come to the end of this image set.")
        st.stop()  # Stop further execution beyond 30 images

    #dense_0, dense_1, dense_2, dense_3

    # Load images based on image_id and index
    #cxr, textured_cxr, lung_noised = load_images(st.session_state.image_id, st.session_state.index)

    # Login logic
    if st.session_state.user is None:
        st.title("Login")
        username = st.selectbox("Select User", USERS)
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = check_auth(username, password)
            if user:
                st.session_state.user = user
                st.success(f"Welcome {user}!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    if st.session_state.user:
        st.title("Density Mapper")

        csv_data = read_csv_from_gcs(st.session_state.user)
        st.session_state.count = int(csv_data[-1][0])+1 if csv_data and len(csv_data) > 1 else 1
        print("count from csv :",st.session_state.count)
        st.session_state.image_id, st.session_state.index = get_image_id_index(st.session_state.count)
        print("starting count, image id, idx", st.session_state.count, st.session_state.image_id, st.session_state.index)

        if (st.session_state.count == 0):
            empty_image = np.zeros((256, 256), dtype=np.uint8)
            cxr = empty_image
            mask = empty_image
            textured_cxr = empty_image
            lung_noised = empty_image

        else:
            # Load images based on image_id and index
            cxr, textured_cxr, lung_noised = load_images(st.session_state.image_id, st.session_state.index)

        # Top row: Original CXR, Noise, Synthetic CXR
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(cxr, caption="Original CXR", width=200)
        with col2:
            st.image(textured_cxr, caption="Noise", width=200)
        with col3:
            st.image(lung_noised, caption="Synthetic CXR", width=200)
        with col4:
            # Save button and progress tracking
            progress = st.slider("Progress", 1, 30, value=st.session_state.count, key="progress_slider", disabled=True)
            label = "Save & Continue"
            if(st.session_state.count == 0):
                label = "Start"
                # Expandable instructions section
            if st.button(label):
                if st.session_state.count > 30:
                    st.info("Processing complete! No further images to process.")
                    return

                print("saving count, image id, idx", st.session_state.count, st.session_state.image_id,
                      st.session_state.index)
                create_csv(st.session_state.user, st.session_state.count, st.session_state.max_thresh0,
                           st.session_state.max_thresh1, st.session_state.max_thresh2)
                st.success(f"Data saved!")

                st.session_state.count += 1
                st.session_state.image_id, st.session_state.index = get_image_id_index(st.session_state.count)
                st.rerun()

            #instructions:
            with st.expander("Instructions: ", expanded=False):
                st.markdown(
                    """
                    <div style="background-color: yellow; padding: 10px; border-radius: 5px;">
                    <ul>
                    <li>Set the brightness of your display to maximum.</li>
                    <li>Synthetic density (middle image in top row) is added to "Original CXR" to obtain "Synthetic CXR".</li>
                    <li>Adjust the brightness thresholds using the sliders provided to obtain the correct density maps for each level of RALE density.</li>
                    <li>If a density level has an absence of pixels at the upper limit, please set the Max Value to 255.</li>
                    <li>Click "Save & continue" to proceed to the next image. The progress is shown in the progress bar.</li>
                    <li>You may close the window and resume the process later when you reopen the window.</li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # Re-render the progress bar with the updated count
                #st.slider("Progress", 1, 30, value=st.session_state.count, key="progress_slider", disabled=True)

        apply_thresholds(cxr, textured_cxr, st.session_state.max_thresh0,
                         st.session_state.max_thresh1,
                         st.session_state.max_thresh2)

        # Horizontal line to separate rows
        st.divider()

        # Bottom row: Density maps with one max slider for each
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cxr_with_dense_0 = cv.add(cxr, st.session_state.dense_0.astype(np.uint8))
            st.image(cxr_with_dense_0, caption="CXR + Density 0", width=200)
            min_density0 = 0
            st.text_input("Min Density 0", value=min_density0, disabled=True)

            # Allow slider to go up to 256, but display max as 255
            max_slider0 = st.slider("Max Density 0", min_density0, 255,
                                    value=st.session_state.max_thresh0, key="max_slider0")

            if max_slider0 != st.session_state.max_thresh0:
                st.session_state.max_thresh0 = max_slider0
                apply_thresholds(cxr, textured_cxr, st.session_state.max_thresh0,
                                 st.session_state.max_thresh1, st.session_state.max_thresh2)
                st.rerun()

            # If Density 0 is set to 255, force the rest to 255
            max_slider1_disabled = max_slider0 >= 255
            max_slider2_disabled = max_slider1_disabled or st.session_state.max_thresh1 >= 255

        with col2:
            cxr_with_dense_1 = cv.add(cxr, st.session_state.dense_1.astype(np.uint8))
            st.image(cxr_with_dense_1, caption="CXR + Density 1", width=200)

            min_density1 = max_slider0
            st.text_input("Min Density 1", value=min_density1, disabled=True)

            max_slider1_value = 255 if max_slider1_disabled else max(st.session_state.max_thresh1, min_density1)
            max_slider1 = st.slider("Max Density 1", min_density1, 256,
                                    value=max_slider1_value, key="max_slider1", disabled=max_slider1_disabled)

            if not max_slider1_disabled and max_slider1 != st.session_state.max_thresh1:
                st.session_state.max_thresh1 = max_slider1
                apply_thresholds(cxr, textured_cxr, st.session_state.max_thresh0,
                                 st.session_state.max_thresh1, st.session_state.max_thresh2)
                st.rerun()

            # Disable next slider if max_slider1 is 255
            max_slider2_disabled = max_slider1 >= 255

        with col3:
            cxr_with_dense_2 = cv.add(cxr, st.session_state.dense_2.astype(np.uint8))
            st.image(cxr_with_dense_2, caption="CXR + Density 2", width=200)

            min_density2 = max_slider1
            st.text_input("Min Density 2", value=min_density2, disabled=True)

            max_slider2_value = 255 if max_slider2_disabled else max(st.session_state.max_thresh2, min_density2)
            max_slider2 = st.slider("Max Density 2", min_density2, 256,
                                    value=max_slider2_value, key="max_slider2", disabled=max_slider2_disabled)

            if not max_slider2_disabled and max_slider2 != st.session_state.max_thresh2:
                st.session_state.max_thresh2 = max_slider2
                apply_thresholds(cxr, textured_cxr, st.session_state.max_thresh0,
                                 st.session_state.max_thresh1, st.session_state.max_thresh2)
                st.rerun()

        with col4:
            cxr_with_dense_3 = cv.add(cxr, st.session_state.dense_3.astype(np.uint8))
            st.image(cxr_with_dense_3, caption="CXR + Density 3", width=200)
            st.text_input("Min Density 3", value=st.session_state.max_thresh2 if not max_slider2_disabled else 255,
                          disabled=True)
            st.text_input("Max Density 3", value=255, disabled=True)

            apply_thresholds(cxr, textured_cxr, st.session_state.max_thresh0,
                             st.session_state.max_thresh1, st.session_state.max_thresh2)


# Run the app
if __name__ == "__main__":
    main()
