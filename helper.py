from ultralytics import YOLO
import streamlit as st
import cv2
import settings
import requests


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path, device = 'gpu')
    return model

last_name = ""  # Initialize to an empty string or None, depending on your needs

def process_detected_objects(res, model):
    global last_name  # Access the global last_name variable
    msg = st.empty()

    detected_objects_set = set()
    try:
        for box in res[0].boxes:
            class_id = int(box.data[0][5])  # Extract class ID
            object_name = model.names[class_id]  # Map class ID to name

            detected_objects_set.add(object_name)

            # Check if the object is different from the last detected object
            if object_name != last_name:
                #empty container to hold the message

                # Generate the description and display it
                description = generate_description(object_name)
                msg.write(f"{object_name} detected. {description}")
                
                
                # Update last_name to the current object_name
                last_name = object_name

    except Exception as ex:
        st.error(f"An error occurred while processing the detected objects: {str(ex)}")


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.

    Returns:
    None
    """
    # Resize and predict
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    res = model.predict(image, conf=conf)

    # Plot detections
    res_plotted = res[0].plot()
    st_frame.image(
        res_plotted,
        caption='Detected Video',
        channels="BGR",
        use_container_width=True
    )

    # Call the process_detected_objects function to handle detection logic
    process_detected_objects(res, model)



def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the YOLOv8 class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption(
        'Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the YOLOv8 class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    if st.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image                                         
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the YOLOv8 class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))



# extract the name of detected object
def get_detected_object_names(boxes, class_names):
    detected_objects = set()
    for box in boxes:
        class_id = int(box.cls)
        # Map class ID to object name
        object_name = class_names[class_id]
        detected_objects.add(object_name)
    return detected_objects


# Function to generate a description
def generate_description(object_name: str) -> str:
    api_url = "http://192.168.1.10:6500/v1/completions"
    payload = {
        "model": "gemma-2-2b-it",
        "prompt": f"Provide a computer engineering description of {object_name} in one sentence.",
        "temperature": 0.6,
        "max_tokens": 50
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        description = data.get('choices', [{}])[0].get('text', "No description available.")
        first_sentence = description.split('.')[0].strip()
        return first_sentence
    except requests.exceptions.RequestException as e:
        # Improved error handling
        error_message = f"Error: {e}"
        if 'response' in locals():
            error_message += f", Response: {response.text}"
        else:
            error_message += ", No response"
        return error_message
