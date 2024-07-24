import sys
import cv2
# import base64
import numpy as np
# from flask import Flask, request, jsonify
import supervision as sv

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# app = Flask(__name__)


# def image_to_base64(image):
#     # Convert image to base64-encoded string
#     _, buffer = cv2.imencode('.png', image)  # Use the image directly without conversion
#     image_base64 = base64.b64encode(buffer).decode('utf-8')
#     return image_base64

# @app.route('/segment', methods=['POST'])
def segment_image(image):
    # image_file = request.files['image']
    # print(image_file)

    # Read and preprocess the image
    # image_stream = image_file.read()
    # image_array = np.frombuffer(image_stream, dtype=np.uint8)
    # original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(original_image.shape)

    black_image = np.zeros(original_image.shape, dtype=np.uint8)

    masks = mask_generator.generate(original_image)
    
    detections = sv.Detections.from_sam(masks)

    # Create a MaskAnnotator without overlaying on the original image
    mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)

    # Visualize the masks
    mask_visualization = mask_annotator.annotate(black_image, detections)
    # print(mask_visualization.shape)

    # Save the mask visualization to a file
    # cv2.imwrite("./test.png", mask_visualization)

    # Convert the color mask image to a base64-encoded string
    # color_mask_base64 = image_to_base64(mask_visualization)

    # Process segmentation results as needed

    return mask_visualization
    #     return jsonify({"result": "Segmentation successful", "segment": mask_visualization.tolist()})
    # except Exception as e:
    #     return jsonify({"error": str(e)})


def capture_video():
    # Open a connection to the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        # cv2.imshow('Video Capture', frame)

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)
        cv2.imshow('RGB', flipped_frame)

        seg_frame = segment_image(flipped_frame)
        cv2.imshow('Segment', seg_frame)


        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sam_checkpoint = "models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing)
        )

    capture_video()

    # app.run(debug=True, port=7000)
