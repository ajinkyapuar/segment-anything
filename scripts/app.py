import sys
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
import supervision as sv

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

app = Flask(__name__)


def image_to_base64(image):
    # Convert image to base64-encoded string
    _, buffer = cv2.imencode('.png', image)  # Use the image directly without conversion
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        image_file = request.files['image']
        # print(image_file)

        # Read and preprocess the image
        image_stream = image_file.read()
        image_array = np.frombuffer(image_stream, dtype=np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # print(original_image.shape)

        black_image = np.zeros(original_image.shape, dtype=np.uint8)

        masks = mask_generator.generate(original_image)
        
        detections = sv.Detections.from_sam(masks)

        # Create a MaskAnnotator without overlaying on the original image
        mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

        # Visualize the masks
        mask_visualization = mask_annotator.annotate(black_image, detections)
        print(mask_visualization.shape)

        # Save the mask visualization to a file
        cv2.imwrite("./test.png", mask_visualization)

        # Convert the color mask image to a base64-encoded string
        color_mask_base64 = image_to_base64(mask_visualization)

        # Process segmentation results as needed

        return jsonify({"result": "Segmentation successful", "color_mask_base64": color_mask_base64})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    sam_checkpoint = "models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    app.run(debug=True)
