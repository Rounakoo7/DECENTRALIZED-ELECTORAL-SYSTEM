from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage.filters import gabor

app = Flask(__name__)

# Helper to read and convert uploaded file to grayscale NumPy image
def read_image_file(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image format")
    return image

# Segment the iris using Hough Circle Transform
def segment_iris(image):
    # Apply a median blur to reduce noise
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
        pupil_circles = np.uint16(np.around(pupil_circles))
        for pupil_circle in pupil_circles[0, :]:
            pupil_center = (pupil_circle[0], pupil_circle[1])  # Circle center
            pupil_radius = pupil_circle[2]  # Circle radius
            cv2.circle(pupil_mask, pupil_center, pupil_radius, 255, -1)  # Draw filled circle on the mask
    else:
        raise ValueError("Pupil not detected")
    # Create a mask for the iris
    iris_mask = np.zeros_like(image)
    iris_center = pupil_center
    iris_radius = pupil_radius + 20
    cv2.circle(iris_mask, iris_center, iris_radius, 255, -1)
    # Create the concentric mask
    mask = iris_mask - pupil_mask
    # Apply the mask to isolate the iris
    segmented_iris = cv2.bitwise_and(image, image, mask=mask)
    return segmented_iris, pupil_center, pupil_radius, iris_center, iris_radius

def normalize_iris(image, pupil_center, pupil_radius, iris_center, iris_radius, radial_res=64, angular_res=512):
    theta = np.linspace(0, 2 * np.pi, angular_res)
    r = np.linspace(0, 1, radial_res)
    r, theta = np.meshgrid(r, theta)
    x_pupil = pupil_center[0] + pupil_radius * np.cos(theta)
    y_pupil = pupil_center[1] + pupil_radius * np.sin(theta)
    x_iris = iris_center[0] + iris_radius * np.cos(theta)
    y_iris = iris_center[1] + iris_radius * np.sin(theta)
    x = (1 - r) * x_pupil + r * x_iris
    y = (1 - r) * y_pupil + r * y_iris
    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)
    normalized = cv2.remap(image, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return normalized.T

def encode_features(normalized_iris):
    # Apply Gabor filter to extract texture
    real, imag = gabor(normalized_iris, frequency=0.1)
    # Binary encoding based on sign
    iris_code = (real > 0).astype(np.uint8)
    return iris_code

def match_iris(code1, code2, threshold=0.445159912109375):
    # XOR and count differing bits
    xor_result = np.bitwise_xor(code1, code2)
    hamming_distance = np.sum(xor_result) / code1.size
    return hamming_distance <= threshold, hamming_distance

@app.route('/match', methods=['POST'])
def match_images():
    try:
        # Read search image and ID
        search_image_file = request.files.get('search_image')
        search_id = request.form.get('search_id')

        if not search_image_file or not search_id:
            return jsonify({'error': 'Missing search_image or search_id'}), 400

        search_image_np = read_image_file(search_image_file)
        search_segmented, search_pupil_center, search_pupil_radius, search_iris_center, search_iris_radius = segment_iris(search_image_np)
        search_normalized = normalize_iris(search_image_np, search_pupil_center, search_pupil_radius, search_iris_center, search_iris_radius)
        search_code = encode_features(search_normalized)

        # Loop through image list
        i = 0
        match_found = False
        min_hamming_distance = 1
        matched_id = None

        while f'images[{i}][id]' in request.form and f'images[{i}][image]' in request.files:
            img_id = request.form.get(f'images[{i}][id]')
            img_file = request.files.get(f'images[{i}][image]')
            img_np = read_image_file(img_file)

            segmented, pupil_center, pupil_radius, iris_center, iris_radius = segment_iris(img_np)
            normalized = normalize_iris(img_np, pupil_center, pupil_radius, iris_center, iris_radius)
            img_code = encode_features(normalized)

            result, hamming_distance = match_iris(img_code, search_code)
            if result and hamming_distance < min_hamming_distance:
                min_hamming_distance = hamming_distance
                matched_id = img_id

            i += 1

        return jsonify({'match': matched_id == search_id})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
