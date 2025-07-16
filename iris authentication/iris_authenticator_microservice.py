from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, numpy as np
from skimage.filters import gabor

app = Flask(__name__)

# Allow CORS from any origin (for development)
CORS(app, resources={r"/*": {"origins": "*"}})


# Helper to read and convert uploaded file to grayscale NumPy image
def image_acquisition_and_preprocessing(image):
    # Reads the raw bytes from the file and converts it into a NumPy array of unsigned 8-bit integers.
    file_bytes = np.frombuffer(image.read(), np.uint8)
    # Decodes the byte array into a grayscale OpenCV image.
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image format")
    # Returns the decoded grayscale image
    return image

# Segment the iris using Hough Circle Transform
def segment_iris(image):
    # Apply a median blur to reduce noise while preserving edges
    blurred = cv2.medianBlur(image, 5)
    # Use Hough Circle Transform to detect pupil
    pupil_circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=200,
        param1=100, 
        param2=30, 
        minRadius=30, 
        maxRadius=65
    )
    # Create a mask for the detected pupil
    pupil_mask = np.zeros_like(image)
    if pupil_circles is not None:
        # Rounds the coordinates and converts them to unsigned 16-bit integers.
        pupil_circles = np.uint16(np.around(pupil_circles))
        # Draws the pupil circle on the mask. -1 means filled circle.
        for pupil_circle in pupil_circles[0, :]:
            pupil_center = (pupil_circle[0], pupil_circle[1])  # Circle center
            pupil_radius = pupil_circle[2]  # Circle radius
            cv2.circle(pupil_mask, pupil_center, pupil_radius, 255, -1)  # Draw filled circle on the mask
    else:
        raise ValueError("Pupil not detected")
    # Create a mask for the iris
    iris_mask = np.zeros_like(image)
    iris_center = pupil_center
    # Assumes the iris is concentric and has a radius 20 pixels larger than the pupil.
    iris_radius = pupil_radius + 20
    # Draws the iris circle (outer boundary).
    cv2.circle(iris_mask, iris_center, iris_radius, 255, -1)
    # Create the concentric mask
    mask = iris_mask - pupil_mask
    # Apply the mask to isolate the iris
    segmented_iris = cv2.bitwise_and(image, image, mask=mask)
    # Returns the segmented iris and the geometry details (centers and radii).
    return segmented_iris, pupil_center, pupil_radius, iris_center, iris_radius

def normalize_iris(image, pupil_center, pupil_radius, iris_center, iris_radius, radial_res=64, angular_res=512):
    # Creates evenly spaced angles from 0 to 2π.
    theta = np.linspace(0, 2 * np.pi, angular_res)
    # Creates a grid of radial and angular values.
    r = np.linspace(0, 1, radial_res)
    r, theta = np.meshgrid(r, theta)
    # Finds coordinates on the pupil boundary for all angles.
    x_pupil = pupil_center[0] + pupil_radius * np.cos(theta)
    y_pupil = pupil_center[1] + pupil_radius * np.sin(theta)
    # Finds coordinates on the iris boundary for all angles.
    x_iris = iris_center[0] + iris_radius * np.cos(theta)
    y_iris = iris_center[1] + iris_radius * np.sin(theta)
    # Interpolates between pupil and iris boundaries for every point on the grid.
    x = (1 - r) * x_pupil + r * x_iris
    y = (1 - r) * y_pupil + r * y_iris
    # Clips coordinates to stay within image bounds.
    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)
    # Samples the original image at the new coordinates to create a normalized rectangular iris image.
    normalized = cv2.remap(image, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    # Returns the transposed version of the normalized image.
    return normalized.T

def feature_extraction(normalized_iris):
    # Apply Gabor filter to extract texture(real and imaginary parts).
    real, imag = gabor(normalized_iris, frequency=0.1)
    # Binarizes the real part: 1 where real > 0, else 0.
    iris_code = (real > 0).astype(np.uint8)
    # Returns the binary iris code.
    return iris_code

def feature_matching(code1, code2):
    # XOR and count differing bits.
    xor_result = np.bitwise_xor(code1, code2)
    # Calculates the normalized Hamming distance.
    hamming_distance = np.sum(xor_result) / code1.size
    # Returns the hamming distance.
    return hamming_distance

# Defines the API endpoint /match which accepts POST requests.
@app.route('/match', methods=['POST'])
def match_images():
    try:
        # Extracts the search image and its associated ID.
        search_image_file = request.files.get('search_image')
        search_id = request.form.get('search_id')

        if not search_image_file or not search_id:
            return jsonify({'error': 'Missing search_image or search_id'}), 400

        # Processes the search image: segment → normalize → encode.
        search_image_np = image_acquisition_and_preprocessing(search_image_file)
        search_segmented, search_pupil_center, search_pupil_radius, search_iris_center, search_iris_radius = segment_iris(search_image_np)
        search_normalized = normalize_iris(search_image_np, search_pupil_center, search_pupil_radius, search_iris_center, search_iris_radius)
        search_code = feature_extraction(search_normalized)

        # Initializes loop variables.
        i = 0
        match_found = False
        min_hamming_distance = 1
        matched_id = None

        # Initializes loop variables and Iterates over array images and ids.
        while f'images[{i}][id]' in request.form and f'images[{i}][image]' in request.files:
            img_id = request.form.get(f'images[{i}][id]')
            img_file = request.files.get(f'images[{i}][image]')
            img_np = image_acquisition_and_preprocessing(img_file)

            # Processes each array image: segment → normalize → encode.
            segmented, pupil_center, pupil_radius, iris_center, iris_radius = segment_iris(img_np)
            normalized = normalize_iris(img_np, pupil_center, pupil_radius, iris_center, iris_radius)
            img_code = feature_extraction(normalized)

            # finds the best match having the minimum hamming distance.
            hamming_distance = feature_matching(img_code, search_code)
            if hamming_distance < min_hamming_distance:
                min_hamming_distance = hamming_distance
                matched_id = img_id

            i += 1

        # Returns whether the matched ID equals the expected search ID.
        return jsonify({'match': matched_id == search_id})

    # Handles unexpected errors.
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)
