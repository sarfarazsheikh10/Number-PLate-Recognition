import numpy as np
import streamlit as st
from numberPlateRecognition import NumberPlateRecognizer
from PIL import Image
import tempfile, cv2


def main():

    # ------------------------- Home page contents ---------------------------------------------------------
    home_txt_0 = st.markdown('<p style="font-size: 42px;">Welcome to my App!</p>', unsafe_allow_html=True)
    home_txt_1 = st.markdown("""
    This __<u>Number Plate Recognition App</u>__ is built using Streamlit and OpenCV.
    
    It uses custom-trained (SOTA) YOLOv4 model for detecting number plates and uses Tesseract OCR to
    read and extract the alphanumerics from the number plates.

    """, unsafe_allow_html=True)

    # Adding sidebar
    st.sidebar.title(':violet[Activity Bar]')

    st.sidebar.markdown("##")
    task = st.sidebar.selectbox(":violet[__Task__]",("Select", "Number Plate Recognition"))
    st.sidebar.markdown("#")
    inputType = st.sidebar.radio(":violet[__Select Input Type__]", ("Image", "Video") )
    
    # -------------------------------------------------------------------------------------------------------

    if task == "Number Plate Recognition" and inputType=="Image" :
        
        # ------------------------------- Texts -------------------------------------------------------------
        home_txt_0.empty()
        home_txt_1.empty()
        st.markdown("# <u>Number Plate Recognition</u>", unsafe_allow_html=True)
        st.markdown("""
        ### Detection for Images
        Upload an image and expect getting an output image with detected bounding boxes and plate numbers.
        """)
        # ---------------------------------------------------------------------------------------------------
        file = st.file_uploader('Upload Image', type = ['jpg','jpeg'], label_visibility = "hidden")

        if file!=None:

            # read image
            image=Image.open(file)
            image= np.array(image)
            st.write("#")
            st.image(image, caption='Original Image')
            st.write("#")
            
            instance = NumberPlateRecognizer()
            outputImage = instance.processImage(image)

            cv2.imwrite("data/detections/detected_image.jpg", cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB) )

            st.image(outputImage, caption='Processed Image')
        

    elif task == "Number Plate Recognition" and inputType=="Video":

        # ------------------------------- Texts -------------------------------------------------------------
        home_txt_0.empty()
        home_txt_1.empty()
        # coming_soon = "<h1 style= 'text-align: center; font-size:75px'>Coming Soon... <p style='font-size:20px; margin-top:25px' ><i>We are still working on it</i></p> </h1>"
        # st.markdown(coming_soon, unsafe_allow_html=True)
        st.markdown("# <u>Number Plate Recognition</u>", unsafe_allow_html=True)
        st.markdown("""
        ### Detection for Videos
        Upload a video and expect getting an output video with detected bounding boxes and plate numbers.
        """)
        # ---------------------------------------------------------------------------------------------------

        file = st.file_uploader('Upload Video', type = ['mp4'], label_visibility = "hidden")

        if file!=None:

            # read video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            
            instance = NumberPlateRecognizer()                      # Creating an object of NumberPlateRecognizer class
            instance.processVideo(tfile.name)                       # processVideo() saves the result as detected_video.mp4

            video_file = open('data/detections/detected_video.mp4', 'rb')
            video_bytes = video_file.read()

            st.video(video_bytes)

        


    elif task == "Select":
        print()


if __name__ == '__main__':
    main()

    



    
