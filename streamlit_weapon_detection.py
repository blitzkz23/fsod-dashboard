# To run use
# $ streamlit run yolor_streamlit_demo.py

from yolor import *
import tempfile
import cv2
import streamlit as st
from PIL import Image
torch.no_grad()

def main():
    DEMO_IMAGE = "assets/images/demo.jpg"
    DEMO_VIDEO = "assets/videos/demo.mp4"

    st.title("Firearms and Sharp Object Detector Dashboard")

    st.markdown(
        """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 350px
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 350px
                margin-left: -350px
            }
            </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("FSOD Sidebar")

    app_page = st.sidebar.selectbox('Choose to Select Page',
                                    ['About App', 'Run on Image', 'Run on Video'])

    if app_page == 'About App':
        st.image('assets/images/gun.jpg')
        st.markdown('''
            Aplikasi ini dibuat untuk mendeteksi adanya ancaman senjata tajam dan senjata api.   &nbsp;&nbsp;Adapun beberapa objek kelas yang dapat dideteksi oleh aplikasi ini sebagai berikut:
            1. Pisau
            2. Kapak
            3. Celurit
            4. Pistol
            5. Senapan Serbu
            6. Bukan Senjata (barang genggaman)
        ''')

    elif app_page == 'Run on Image':
        st.sidebar.subheader("Settings")

        st.sidebar.markdown('---')
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.5)
        st.sidebar.markdown('---')

        # Checkboxes
        enable_GPU = st.sidebar.checkbox("Enable GPU")
        custom_classes = st.sidebar.checkbox("Custom Classes")
        st.sidebar.markdown('---')
        assigned_class_id = []

        # Custom classes
        if custom_classes:
            assigned_class = st.sidebar.multiselect("Select The Custom Classes", list(names), default="pistol")
            for each in assigned_class:
                assigned_class_id.append(names.index(each))

        # Uploading Images
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", ["jpg", "jpeg", "png"])
        tffile = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            tffile.write(img_file_buffer.getvalue())
        else:
            demo_image = DEMO_IMAGE
            tffile.name = demo_image
            image = np.array(Image.open(demo_image))

        print(tffile.name)

        st.sidebar.text("Original Image")
        st.sidebar.image(image)

        stframe = st.empty()
        st.sidebar.markdown('---')

        st.markdown(f"<h2 align=center>Detected Objects</h2>", unsafe_allow_html=True)
        kpi1_text = st.markdown(f"<h3 align=center>0</h3>", unsafe_allow_html=True)

        # Calling YOLOR
        with torch.no_grad():
            detect_img(tffile.name, enable_GPU, confidence, assigned_class_id, stframe,
                   kpi1_text)

    else:
        st.sidebar.subheader("Settings")

        st.sidebar.markdown('---')
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.25)
        st.sidebar.markdown('---')

        # Checkboxes
        enable_GPU = st.sidebar.checkbox("Enable GPU")
        custom_classes = st.sidebar.checkbox("Custom Classes")
        assigned_class_id = []

        enable_webcam = st.sidebar.button("Use Webcam")

        st.sidebar.markdown('---')

        # Custom classes
        if custom_classes:
            assigned_class = st.sidebar.multiselect("Select The Custom Classes", list(names), default="pistol")
            for each in assigned_class:
                assigned_class_id.append(names.index(each))

        # Uploading video
        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "asf", "m4v", "mkv"])
        tffile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

        vid = cv2.VideoCapture(DEMO_VIDEO)

        # Get input video
        if not video_file_buffer:
            if enable_webcam:
                vid = cv2.VideoCapture(0)
            else:
                tffile.name = DEMO_VIDEO
                dem_vid = open(tffile.name, 'rb')
                demo_bytes = dem_vid.read()

                st.sidebar.text("Input Video")
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())
            dem_vid = open(tffile.name, 'rb')
            demo_bytes = dem_vid.read()

            st.sidebar.text("Input Video")
            st.sidebar.video(demo_bytes)

        print(tffile.name)

        stframe = st.empty()
        st.sidebar.markdown('---')

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Tracked Objects**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Width**")
            kpi3_text = st.markdown("0")

        # Calling YOLOR + DeepSORT
        with torch.no_grad():
            if enable_webcam:
                load_yolor_and_process_each_frame_webcam('0', enable_GPU, confidence, assigned_class_id, kpi1_text,
                                                  kpi2_text,
                                                  kpi3_text, stframe)
            else:
                load_yolor_and_process_each_frame_video(tffile.name, enable_GPU, confidence, assigned_class_id, kpi1_text,
                                                  kpi2_text,
                                                  kpi3_text, stframe)
        st.text("Video is Processed")
        vid.release()

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass