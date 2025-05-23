'''
This script performs object detection on an image using the Grounding DINO model.
Then, it detects circles in the image and filters them based on their position relative to the detected bounding boxes.
'''

from PIL import Image
import requests
import cv2
import numpy as np
import math
from transformers import GroundingDinoProcessor
from transformers import GroundingDinoForObjectDetection

import torch
import matplotlib.pyplot as plt

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes, text, output_path):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label = f'{text}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(output_path)

def run_inference(image, text):
    processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    inputs = processor(images=image, text=preprocess_caption(text), return_tensors="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
    model = model.to(device)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Using CPU for inference.")

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs, processor, device

def detect_and_filter_circles(image_path, boxes):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    blurred_img = cv2.GaussianBlur(img, (0, 0), 1)

    circles = cv2.HoughCircles(blurred_img, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1,
                               minDist=20,
                               param1=200,
                               param2=20,
                               minRadius=10,
                               maxRadius=15)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print("Detected circle centers (x, y) and radius:")
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Check if center lies within any of the bounding boxes
            inside_box = any(
                xmin <= center[0] <= xmax and ymin <= center[1] <= ymax
                for (xmin, ymin, xmax, ymax) in boxes
            )
            if inside_box:
                print(f"Center: {center}, Radius: {radius}")
                theta = math.radians(0)
                x = int(center[0] + radius * math.cos(theta))
                y = int(center[1] + radius * math.sin(theta))
                cv2.circle(color, center, 1, (255, 0, 255), 1)
                print(f"Point on circle at angle 0 degrees: ({x}, {y})")
            else:
                print(f"Circle at {center} is outside bounding boxes, discarded.")
    else:
        print("No circles detected.")

    cv2.imwrite('/home/jypaulsung/Sapien/detected_circles.png', color) # Save the image at the specified path, change as needed

if __name__ == "__main__":
    image_path = "/home/jypaulsung/Sapien/only_coke_1.png" # Change to your image path
    image = Image.open(image_path).convert("RGB")
    text = "coke can"

    outputs, processor, device = run_inference(image, text)

    width, height = image.size
    postprocessed_outputs = processor.image_processor.post_process_object_detection(
        outputs,
        target_sizes=[(height, width)],
        threshold=0.3
    )
    results = postprocessed_outputs[0]

    boxes = []
    for score, box in zip(results['scores'], results['boxes']):
        xmin, ymin, xmax, ymax = box.tolist()
        boxes.append((xmin, ymin, xmax, ymax))
        print(f"Score: {score:.2f}, Box: ({xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f})")
        print(f"Center: ({(xmin+xmax)/2:.1f}, {(ymin+ymax)/2:.1f})")

    plot_results(image, results['scores'].tolist(), results['labels'].tolist(),
                 results['boxes'].tolist(), text, "/home/jypaulsung/Sapien/coke_detection_1.png")

    detect_and_filter_circles(image_path, boxes)
