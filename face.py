import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifiers for face and eye detection.
# These files are typically located in the OpenCV data directory.
# You might need to provide the full path if they're not found.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# A detailed dataset for skin tone and face shape analysis with makeup recommendations.
# This data structure maps skin tone and face shape to specific cosmetic products and colors.
MAKEUP_RECOMMENDATIONS = {
    "cool_tones": {
        "foundation": "Cool-toned, rosy, or neutral shades. Avoid yellow-based foundations.",
        "blush": "Shades of pink, mauve, or berry. Avoid peach or orange.",
        "contour": "Cool, taupe-based shades. Avoid orange or red undertones."
    },
    "warm_tones": {
        "foundation": "Warm-toned, yellow-based, or golden shades. Avoid pink or red undertones.",
        "blush": "Shades of peach, coral, or bronze. Avoid pink or berry.",
        "contour": "Warm, bronze-based shades. Avoid gray or taupe undertones."
    },
    "neutral_tones": {
        "foundation": "Neutral shades that balance yellow and pink undertones. Most colors will suit you.",
        "blush": "A wide range of shades including soft pinks, peaches, and berries.",
        "contour": "Neutral, medium brown shades."
    },
    "face_shapes": {
        "Circle": {
            "contour": "Apply to the sides of the forehead and jawline to create angles.",
            "blush": "Apply to the apples of the cheeks and blend upwards towards the temples."
        },
        "Round/Wide": {
            "contour": "Apply to the temples and cheekbones to add definition.",
            "blush": "Apply below the cheekbones to create a slimmer appearance."
        },
        "Long/Narrow": {
            "contour": "Apply to the chin and hairline to shorten the face.",
            "blush": "Apply horizontally on the apples of the cheeks to add width."
        },
        "Oval": {
            "contour": "Minimal contouring is needed. Apply lightly under the cheekbones.",
            "blush": "Apply to the apples of the cheeks."
        },
        "Undetermined": {
            "contour": "Apply to the temples and cheekbones to add definition.",
            "blush": "Apply to the apples of the cheeks."
        }
    }
}

# A dataset of colors grouped by tone to serve as a recommendation pool.
# Each entry is now a tuple of (color_name, BGR_values).
COLOR_DATASET = {
    "cool_tones": [
        ("Maroon", (128, 0, 0)),
        ("DarkRed", (139, 0, 0)),
        ("Red", (255, 0, 0)),
        ("Navy", (0, 0, 128)),
        ("Purple", (128, 0, 128)),
        ("Magenta", (255, 0, 255)),
        ("Cyan", (0, 255, 255)),
        ("Yellow", (255, 255, 0)),
        ("Pink", (255, 192, 203)),
        ("LightBlue", (173, 216, 230)),
        ("Teal", (128, 128, 0)),
        ("Olive", (0, 128, 128)),
        ("BlueViolet", (138, 43, 226)),
    ],
    "warm_tones": [
        ("Green", (0, 128, 0)),
        ("Lime", (0, 255, 0)),
        ("SpringGreen", (0, 255, 127)),
        ("Orange", (0, 165, 255)),
        ("OrangeRed", (0, 165, 255)),
        ("Gold", (0, 215, 255)),
        ("Chartreuse", (127, 255, 0)),
        ("PaleGreen", (152, 251, 152)),
        ("Crimson", (128, 0, 255)),
        ("Blue", (0, 0, 255)),
        ("DarkOrange", (255, 140, 0)),
        ("Tomato", (255, 99, 71)),
        ("Beige", (245, 245, 220)),
    ],
    "neutral_tones": [
        ("Silver", (192, 192, 192)),
        ("Gray", (128, 128, 128)),
        ("Black", (0, 0, 0)),
        ("White", (255, 255, 255)),
        ("LightGray", (211, 211, 211)),
        ("DarkGray", (169, 169, 169)),
        ("Khaki", (240, 230, 140)),
        ("OliveGreen", (128, 128, 0)),
        ("SaddleBrown", (19, 69, 139)),
        ("Tan", (210, 180, 140)),
        ("Brown", (139, 69, 19)),
        ("CornflowerBlue", (100, 149, 237)),
        ("Bisque", (255, 228, 196)),
    ]
}


def get_average_color(image_roi):
    """
    Calculates the average color of a given region of interest (ROI).

    Args:
        image_roi (numpy.ndarray): The region of interest, a NumPy array of pixels.

    Returns:
        tuple: An average BGR color tuple (e.g., (120, 150, 180)).
    """
    # Reshape the image to be a list of pixels.
    pixels = image_roi.reshape(-1, 3)
    # Calculate the mean for each BGR channel and cast to integer.
    avg_color_bgr = np.mean(pixels, axis=0).astype(int)
    # Explicitly convert the NumPy array to a list to ensure the correct format for OpenCV.
    return tuple(avg_color_bgr.tolist())


def get_recommended_colors(bgr_color):
    """
    Recommends colors based on the warmth of the detected BGR skin tone.

    Args:
        bgr_color (tuple): The average BGR color of the detected face.

    Returns:
        list: A list of recommended color tuples (color_name, BGR_values).
    """
    b, g, r = bgr_color

    # Simple heuristic to determine warmth of the skin tone.
    # Higher red and green values suggest a warmer tone.
    if r > 150 and g > 100 and b < 100:
        return COLOR_DATASET["cool_tones"]
    # Cooler tones have higher blue and less red/green.
    elif b > r and b > g:
        return COLOR_DATASET["warm_tones"]
    # Neutral tones are a mix of all colors.
    else:
        return COLOR_DATASET["neutral_tones"]


def get_skin_tone_type(bgr_color):
    """
    Returns the skin tone type (warm, cool, or neutral).
    """
    b, g, r = bgr_color
    if r > 150 and g > 100 and b < 100:
        return "cool_tones"
    elif b > r and b > g:
        return "warm_tones"
    else:
        return "neutral_tones"


def get_eye_color(eye_roi):
    """
    Analyzes the eye region and returns the detected eye color.

    Args:
        eye_roi (numpy.ndarray): The region of interest containing the eye.

    Returns:
        str: The detected eye color (e.g., "Blue", "Green", "Brown") or "Undetermined".
    """
    # Convert the eye ROI to HSV color space for better color detection.
    hsv_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for common eye colors.
    # Hue range for blue: 90-130
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])

    # Hue range for green: 40-80
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])

    # Hue range for brown: 10-25
    brown_lower = np.array([10, 50, 50])
    brown_upper = np.array([25, 255, 255])

    # Create masks for each color.
    blue_mask = cv2.inRange(hsv_roi, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)
    brown_mask = cv2.inRange(hsv_roi, brown_lower, brown_upper)

    # Calculate the number of pixels for each color.
    blue_pixels = cv2.countNonZero(blue_mask)
    green_pixels = cv2.countNonZero(green_mask)
    brown_pixels = cv2.countNonZero(brown_mask)

    # Find the color with the most pixels.
    color_pixels = {
        "Blue": blue_pixels,
        "Green": green_pixels,
        "Brown": brown_pixels,
    }

    # Simple threshold to avoid false positives.
    if max(color_pixels.values()) > 100:
        return max(color_pixels, key=color_pixels.get)
    else:
        return "Undetermined"


def main():
    """
    Main function to run the face detection, color sampling, and basic shape analysis on a live webcam feed.
    """
    # Initialize the webcam. The argument '0' typically refers to the default camera.
    # If you have multiple cameras, you might need to try '1', '2', etc.
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully.
    if not cap.isOpened():
        print(
            "Error: Could not open webcam. Please check if the camera is connected and not in use by another application.")
        return

    print("Webcam is now active. Press 'q' to quit.")

    while True:
        # Capture a frame from the webcam.
        ret, frame = cap.read()

        # If the frame was not captured successfully, break the loop.
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Convert the frame to grayscale for faster face detection.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame.
        # Adjusted minSize to ensure larger faces are also considered for analysis.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Create a blank black image to display the makeup recommendations on.
        # This prevents the text from overlapping the video feed.
        info_height = 200
        total_height = frame.shape[0] + info_height
        total_frame = np.zeros((total_height, frame.shape[1], 3), dtype=np.uint8)

        # Copy the webcam feed to the top part of the combined frame.
        total_frame[0:frame.shape[0], 0:frame.shape[1]] = frame

        # Iterate through each detected face.
        for i, (x, y, w, h) in enumerate(faces):
            # Define a larger, central region of interest (ROI) within the face
            # to sample the skin tone, avoiding hair or shadows at the edges.
            # Adjusted ROI to cover more of the face for better color and shape analysis.
            roi_y_start = y + int(h * 0.2)
            roi_y_end = y + int(h * 0.8)
            roi_x_start = x + int(w * 0.2)
            roi_x_end = x + int(w * 0.8)

            face_roi = total_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            # Define ROI for eyes within the face ROI.
            # This helps improve eye detection accuracy.
            eyes_y_start = y + int(h * 0.2)
            eyes_y_end = y + int(h * 0.45)
            eyes_roi = total_frame[eyes_y_start:eyes_y_end, x:x + w]

            # Use a try-except block to handle potential errors.
            try:
                # Get the average color of the sampled face ROI.
                if face_roi.size > 0:
                    avg_color_bgr = get_average_color(face_roi)

                    # Add a check to ensure avg_color_bgr is a valid BGR tuple.
                    if isinstance(avg_color_bgr, tuple) and len(avg_color_bgr) == 3:
                        # Draw a rectangle around the detected face and a color swatch.
                        cv2.rectangle(total_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        color_swatch_height = 50
                        cv2.rectangle(total_frame, (x, y - color_swatch_height), (x + w, y), avg_color_bgr, -1)

                        # --- COLOR RECOMMENDATION LOGIC ---
                        # Get a list of recommended colors.
                        recommended_colors = get_recommended_colors(avg_color_bgr)

                        # Display the first three recommended colors.
                        for i, (rec_name, rec_bgr) in enumerate(recommended_colors[:3]):
                            rec_x = x + (i * (w // 3))

                            # Determine if text should be black or white for contrast.
                            rec_b, rec_g, rec_r = rec_bgr
                            rec_brightness = (rec_r * 0.299 + rec_g * 0.587 + rec_b * 0.114)
                            text_color = (0, 0, 0) if rec_brightness > 128 else (255, 255, 255)

                            cv2.rectangle(total_frame, (rec_x, y - color_swatch_height - 30),
                                          (rec_x + (w // 3), y - color_swatch_height), rec_bgr, -1)
                            cv2.putText(total_frame, rec_name, (rec_x + 5, y - color_swatch_height - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                        # Update the color text to include the recommended color.
                        color_text = f'BGR: {avg_color_bgr}'
                        text_color_on_swatch = (0, 0, 0) if (avg_color_bgr[2] * 0.299 + avg_color_bgr[1] * 0.587 +
                                                             avg_color_bgr[0] * 0.114) > 128 else (255, 255, 255)
                        cv2.putText(total_frame, color_text, (x + 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    text_color_on_swatch, 2)

                        # --- BASIC FACE SHAPE ANALYSIS ---
                        # A simple heuristic based on the aspect ratio (width/height) of the face box.
                        aspect_ratio = w / h
                        face_shape = "Undetermined"

                        # Check for a circular shape (aspect ratio close to 1).
                        if 0.9 <= aspect_ratio <= 1.1:
                            face_shape = "Circle"
                        # Check for a round/wide shape.
                        elif aspect_ratio > 1.1:
                            face_shape = "Round/Wide"
                        # Check for a long/narrow shape.
                        elif aspect_ratio < 0.9:
                            if aspect_ratio < 0.8:
                                # A very narrow face might be considered a long/narrow shape.
                                face_shape = "Long/Narrow"
                            else:
                                # A more typical long face with a good balance.
                                face_shape = "Oval"

                        # Display the detected face shape.
                        cv2.putText(total_frame, f'Shape: {face_shape}', (x + 10, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2)

                        # --- EYE COLOR ANALYSIS ---
                        # Detect eyes within the eye_roi of the face.
                        eyes = eye_cascade.detectMultiScale(eyes_roi)

                        if len(eyes) > 0:
                            # Use the first detected eye for simplicity.
                            (ex, ey, ew, eh) = eyes[0]
                            eye_roi_final = eyes_roi[ey:ey + eh, ex:ex + ew]

                            # Get the eye color.
                            eye_color = get_eye_color(eye_roi_final)
                        else:
                            eye_color = "Undetermined"

                        # Display the detected eye color.
                        cv2.putText(total_frame, f'Eye Color: {eye_color}', (x + 10, y + h + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # --- DETAILED MAKEUP RECOMMENDATIONS ---
                        # Get skin tone and face shape type.
                        skin_tone_type = get_skin_tone_type(avg_color_bgr)

                        # Get makeup recommendations based on the detected attributes.
                        foundation_rec = MAKEUP_RECOMMENDATIONS[skin_tone_type]["foundation"]
                        blush_rec = MAKEUP_RECOMMENDATIONS[skin_tone_type]["blush"]
                        contour_rec = MAKEUP_RECOMMENDATIONS[skin_tone_type]["contour"]

                        blush_shape_rec = MAKEUP_RECOMMENDATIONS["face_shapes"][face_shape]["blush"]
                        contour_shape_rec = MAKEUP_RECOMMENDATIONS["face_shapes"][face_shape]["contour"]

                        # Display recommendations in the info panel below the video feed.
                        text_start_y = frame.shape[0] + 20
                        text_x = 10
                        line_spacing = 30

                        cv2.putText(total_frame, "Recommended Cosmetics:", (text_x, text_start_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        cv2.putText(total_frame, f'Foundation: {foundation_rec}', (text_x, text_start_y + line_spacing),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        cv2.putText(total_frame, f'Blush: {blush_rec}', (text_x, text_start_y + 2 * line_spacing),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        cv2.putText(total_frame, f'Contour: {contour_rec}', (text_x, text_start_y + 3 * line_spacing),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        cv2.putText(total_frame, f'Blush Application: {blush_shape_rec}',
                                    (text_x, text_start_y + 4 * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1)

                        cv2.putText(total_frame, f'Contour Application: {contour_shape_rec}',
                                    (text_x, text_start_y + 5 * line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1)

                    else:
                        cv2.rectangle(total_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(total_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            except Exception as e:
                print(f"Error processing face ROI: {e}")
                cv2.rectangle(total_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the resulting frame.
        cv2.imshow('Face Recognition', total_frame)

        # Break the loop when 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
