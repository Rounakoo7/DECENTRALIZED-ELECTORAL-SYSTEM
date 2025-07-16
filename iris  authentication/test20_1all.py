import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
from skimage.morphology import skeletonize
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
    # Create a mask for the iris
    iris_mask = np.zeros_like(image)
    iris_center = pupil_center
    iris_radius = pupil_radius + 20
    cv2.circle(iris_mask, iris_center, iris_radius, 255, -1)  # Draw filled circle on the mask
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

def remove_eyelashes(image):    
    # Step 1: Use black-hat morphological operation to highlight dark fine structures (eyelashes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))  # wide kernel for horizontal lashes
    tophat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    # Step 2: Threshold the top-hat result to detect strong eyelash regions
    _, eyelash_mask = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)
    # Step 3: Refine mask with morphological closing to fill small gaps
    refined_mask = cv2.morphologyEx(eyelash_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Step 4: Inpaint eyelash regions while preserving iris details
    inpainted_img = cv2.inpaint(image, refined_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    return inpainted_img

def enhance_iris(normalized_iris):
    # Step 1: Histogram Equalization (improves contrast)
    enhanced = cv2.equalizeHist(normalized_iris)

    # Step 2: CLAHE (better localized contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(enhanced)

    # Step 3: Sharpening with a kernel
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    enhanced = cv2.filter2D(enhanced, -1, sharpening_kernel)

    return enhanced


# 4. Encode Iris with Gabor Filters
def encode_iris_with_gabor(normalized_iris, frequencies=[0.1], orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    iris_code = []
    for freq in frequencies:
        for theta in orientations:
            real, imag = gabor(normalized_iris, frequency=freq, theta=theta)
            phase = np.arctan2(imag, real)
            quadrant = np.zeros_like(phase, dtype=np.uint8)
            quadrant[(phase > -np.pi) & (phase <= -np.pi/2)] = 0
            quadrant[(phase > -np.pi/2) & (phase <= 0)] = 1
            quadrant[(phase > 0) & (phase <= np.pi/2)] = 2
            quadrant[(phase > np.pi/2) & (phase <= np.pi)] = 3
            bits_0 = (quadrant >> 1) & 1
            bits_1 = quadrant & 1
            binary_code = np.stack((bits_0, bits_1), axis=-1).reshape(-1)
            iris_code.append(binary_code)
    iris_code = np.concatenate(iris_code).flatten()
    return iris_code[:2048] if len(iris_code) > 2048 else np.pad(iris_code, (0, 2048 - len(iris_code)), 'constant')

# 5. Hamming Distance for Matching
def hamming_distance(code1, code2):
    assert len(code1) == len(code2), "Iris codes must be the same length."
    return np.sum(code1 != code2) / len(code1)

def plot_segmentation(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    print("Checking performance. ....")
    innerfiles = os.listdir(directory_path)
    count = 1
    for innerfile in innerfiles:
        innerfilepath = os.path.join(directory_path, innerfile)
        rootfiles = os.listdir(innerfilepath)
        imagefiles = [f for f in rootfiles if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not imagefiles:
            print(f"No image files found in '{innerfilepath}'.")
            break
        for imagefile in imagefiles:
            imagefilepath = os.path.join(innerfilepath, imagefile) 
            image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE)
            segmented_iris, pupil_center, pupil_radius, iris_center, iris_radius = segment_iris(image)
            normalized_iris = normalize_iris(image, pupil_center, pupil_radius, iris_center, iris_radius)
            
            image2 = cv2.imread("D:\\fingerprint\iris\casia-iris\CASIA1\\2\\002_1_3.jpg", cv2.IMREAD_GRAYSCALE)
            segmented_iris2, pupil_center2, pupil_radius2, iris_center2, iris_radius2 = segment_iris(image2)
            normalized_iris2 = normalize_iris(image2, pupil_center2, pupil_radius2, iris_center2, iris_radius2)
            encode_features(normalized_iris)
            x,y = match_iris( encode_features(normalized_iris),encode_features(normalized_iris2))
            # Plotting with keypress detection
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 1, 1)
            plt.imshow(encode_features(normalized_iris2), cmap='gray')
            plt.title(imagefile)
            plt.axis('off')


            plt.show(block=False)
            plt.pause(100000)
            plt.close()
            print(count, imagefile)
            count+=1

def image_enhance(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def extract_minutiae(img):
    enhanced = image_enhance(img)
    _, binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    binary = (binary == 0).astype(np.uint8)  # invert
    skeleton = skeletonize(binary).astype(np.uint8)

    minutiae = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 1:
                neighbors = skeleton[y-1:y+2, x-1:x+2].flatten()
                count = np.sum(neighbors) - 1
                if count == 1 or count == 3:
                    minutiae.append((x, y))
    return minutiae, skeleton * 255

def match_minutiae(minutiae1, minutiae2, tolerance=15):
    matches = 0
    for (x1, y1) in minutiae1:
        for (x2, y2) in minutiae2:
            if abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance:
                matches += 1
                break
    return matches

def authenticate(img1, img2, threshold=166):

    minutiae1, skeleton1 = extract_minutiae(img1)
    minutiae2, skeleton2 = extract_minutiae(img2)

    matches = match_minutiae(minutiae1, minutiae2)
    if(matches >= threshold):
        return True, matches
    else:
        return False, matches 

def encode_features(normalized_iris):
    # Apply Gabor filter to extract texture
    real, imag = gabor(normalized_iris, frequency=0.1)
    # Binary encoding based on sign
    iris_code = (real > 0).astype(np.uint8)
    return iris_code


def match_iris(code1, code2, threshold = 1):
    # XOR and count differing bits
    xor_result = np.bitwise_xor(code1, code2)
    hamming_distance = np.sum(xor_result) / code1.size
    if(hamming_distance <= threshold):
        return xor_result, hamming_distance
    else:
        return False, hamming_distance


def check_performance(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    print("Checking performance. ....")
    innerfiles1 = os.listdir(directory_path)
    normalized_iris1_array = []
    for innerfile1 in innerfiles1:
        innerfilepath1 = os.path.join(directory_path, innerfile1)
        rootfiles1 = os.listdir(innerfilepath1)
        imagefiles1 = [f for f in rootfiles1 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not imagefiles1:
            print(f"No image files found in '{innerfilepath1}'.")
            break
        for imagefile1 in imagefiles1:
            if(imagefile1[4:5] != "1"):
                continue
            imagefilepath1 = os.path.join(innerfilepath1, imagefile1) 
            image1 = cv2.imread(imagefilepath1, cv2.IMREAD_GRAYSCALE)
            segmented_iris1, pupil_center1, pupil_radius1, iris_center1, iris_radius1 = segment_iris(image1)
            normalized_iris1 = normalize_iris(image1, pupil_center1, pupil_radius1, iris_center1, iris_radius1)
            normalized_iris1_array.append([imagefile1, normalized_iris1])
            print(imagefile1)
    count = 1
    correct = 0
    incorrect = 0
    notfound = 0
    minincorrect = 1
    maxcorrect = 0
    matches_array = []
    innerfiles2 = os.listdir(directory_path)
    for innerfile2 in innerfiles2:
        innerfilepath2 = os.path.join(directory_path, innerfile2)
        rootfiles2 = os.listdir(innerfilepath2)
        imagefiles2 = [f for f in rootfiles2 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not imagefiles2:
            print(f"No image files found in '{innerfilepath2}'.")
            break
        for imagefile2 in imagefiles2:
            if(imagefile2[4:5] == "1"):
                continue
            imagefilepath2 = os.path.join(innerfilepath2, imagefile2) 
            image2 = cv2.imread(imagefilepath2, cv2.IMREAD_GRAYSCALE)
            segmented_iris2, pupil_center2, pupil_radius2, iris_center2, iris_radius2 = segment_iris(image2)
            normalized_iris2 = normalize_iris(image2, pupil_center2, pupil_radius2, iris_center2, iris_radius2)
            current_matches = []
            min_hamming_distance = 1
            second_min_hamming_distance = 1
            min_img_name = ""
            second_min_img_name = ""
            for normalized_iris1 in normalized_iris1_array:
                result, hamming_distance = match_iris(encode_features(normalized_iris1[1]), encode_features(normalized_iris2))
                if(result):
                    if(hamming_distance < min_hamming_distance):
                        second_min_hamming_distance = min_hamming_distance
                        second_min_img_name = min_img_name
                        min_hamming_distance = hamming_distance
                        min_img_name = normalized_iris1[0]
                    elif(hamming_distance < second_min_hamming_distance and hamming_distance != min_hamming_distance):
                        second_min_hamming_distance = hamming_distance
                        second_min_img_name = normalized_iris1[0]    
                    current_matches.append([normalized_iris1[0], hamming_distance])
            if(len(current_matches) == 0):
                notfound+=1
            elif(len(current_matches) >= 1):
                if(min_img_name[0:3] == imagefile2[0:3]):
                    correct+=1
                    if(min_hamming_distance > maxcorrect):
                        maxcorrect = min_hamming_distance
                    if(second_min_hamming_distance < minincorrect and second_min_img_name[0:3] != imagefile2[0:3]):
                        minincorrect = second_min_hamming_distance
                else:
                    incorrect+=1
                    if(min_hamming_distance < minincorrect):
                        minincorrect = min_hamming_distance
            matches_array.append([imagefile2, current_matches])
            print(count, "/432 ", imagefile2, min_img_name, maxcorrect, minincorrect)
            print(correct, incorrect, notfound)
            count+=1
    print(correct, incorrect, notfound, maxcorrect, minincorrect)
    print(matches_array)
    #with open("results.txt", "a") as f:
     #   f.write("")


plot_segmentation("D:\\fingerprint\iris\casia-iris\CASIA1")
#check_performance("D:\\fingerprint\iris\casia-iris\CASIA1")
'''
image1 = cv2.imread("D:\\fingerprint\iris\casia-iris\CASIA1\\2\\002_1_1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("D:\\fingerprint\iris\casia-iris\CASIA1\\2\\002_1_3.jpg", cv2.IMREAD_GRAYSCALE)
segmented_iris1, pupil_center1, pupil_radius1, iris_center1, iris_radius1 = segment_iris(image1)
normalized_iris1 = normalize_iris(image1, pupil_center1, pupil_radius1, iris_center1, iris_radius1)
segmented_iris2, pupil_center2, pupil_radius2, iris_center2, iris_radius2 = segment_iris(image2)
normalized_iris2 = normalize_iris(image2, pupil_center2, pupil_radius2, iris_center2, iris_radius2)

print(match_iris(encode_features(normalized_iris1), encode_features(normalized_iris2)))
#print(hamming_distance(encode_iris_with_gabor(enhance_iris(normalized_iris1)), encode_iris_with_gabor(enhance_iris(normalized_iris2))))
'''