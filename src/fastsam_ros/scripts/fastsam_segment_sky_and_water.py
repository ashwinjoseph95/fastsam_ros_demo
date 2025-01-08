#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from fastsam import FastSAM, FastSAMPrompt
import torch
from PIL import Image as PILImage
import time

class FastSAMSegmenter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('fastsam_segmenter_for_sky_and_water', anonymous=True)
        
        # Setup the ROS subscriber for compressed images
        self.image_sub = rospy.Subscriber('/zed2i/zed_node/left/image_rect_color/compressed', CompressedImage, self.image_callback)
        
        # Setup the ROS publisher for segmented images
        self.image_pub_seg_image = rospy.Publisher('/fastsam/segmented_image', Image, queue_size=10)
        self.image_pub_seg_mask = rospy.Publisher('/fastsam/segmented_mask', Image, queue_size=10)
        
        # Initialize the bridge and FastSAM model
        self.bridge = CvBridge()
        self.model = FastSAM('/home/knadmin/Ashwin/vision_language_model_study/FastSAM/weights/FastSAM.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_count = 0  

        rospy.loginfo("FastSAM Node Initialized.")

    def image_callback(self, msg):
        try:

            # Skip frames (e.g., process every 5th frame)
            self.frame_count += 1
            if self.frame_count % 20 != 0:
                return  # Skip this frame

            # Convert the compressed image to a CV image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo(f"cv_image type: {type(cv_image)}, shape: {cv_image.shape}")


            # cv_image = self.resize_image(self,cv_image)
            # rospy.loginfo(f"Image shape: {cv_image.shape} ")

            # Log the resized image's shape
            rospy.loginfo(f"Post-resize image shape: {cv_image.shape}")


            # Convert the image to PIL format for FastSAM
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Record the start time for inference
            start_time = time.time()

            # Perform segmentation using FastSAM
            # everything_results = self.model(
            #     pil_image,
            #     device=self.device,
            #     retina_masks=True,
            #     imgsz=512,
            #     conf=0.4,
            #     iou=0.9,
            # )

            # Perform segmentation using FastSAM
            everything_results = self.model(
                pil_image,
                device=self.device,
                retina_masks=True,
                imgsz=640,  # Reduce from 1024
                conf=0.3,   # Reduce from 0.4
                iou=0.7,    # Reduce from 0.9
            )

            # Record the end time and compute inference duration
            inference_time = time.time() - start_time
            rospy.loginfo(f"Inference time per image: {inference_time:.3f} seconds")

            # Process the segmentation results
            prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)

            prompts = ['sky', 'water']

            ann = prompt_process.text_prompt(text='sky')  # Segmentation prompt, e.g., "sky"

            # Create the final segmented image
            segmented_image_sky, mask_sky = self.process_masks(ann, cv_image)

            ann = prompt_process.text_prompt(text='water')  # Segmentation prompt, e.g., "sky"

            # Create the final segmented image
            segmented_image_water, mask_water = self.process_masks(ann, cv_image)

            result_image, combined_mask = self.combine_masks_and_extract(cv_image,mask_sky,mask_water)

            # Publish the segmented image
            self.publish_segmented_images(result_image, combined_mask)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def combine_masks_and_extract(self, original_image, mask1, mask2):
        """
        Combines two binary masks using logical OR and applies the resultant mask
        to an image, showing only the parts of the image corresponding to white regions in the mask.

        Args:
        - original_image: The input image (NumPy array in BGR format).
        - mask1: First binary mask (NumPy array with 0 for black, 255 for white).
        - mask2: Second binary mask (NumPy array with 0 for black, 255 for white).

        Returns:
        - result_image: The resultant image with only the regions from the original image
                        corresponding to white regions in the combined mask.
        """
        # Ensure both masks are binary and the same size
        combined_mask = np.logical_and(mask1 > 0, mask2 > 0).astype(np.uint8) * 255  # Resultant binary mask

        # Initialize a blank image with the same shape as the original image
        result_image = np.zeros_like(original_image)

        # Use the combined mask to select parts of the original image
        result_image[combined_mask == 255] = original_image[combined_mask == 255]

        return result_image, combined_mask

    def process_masks(self, ann, cv_image):
        """
        Process segmentation results and create the final image and a binary mask:
        - Masked regions (e.g., "sky") are black in the segmented image.
        - Non-masked regions are white in the binary mask.
        """
        try:
            # Resize the mask to match the original image dimensions
            mask_resized = PILImage.fromarray(ann[0]).resize(
                (cv_image.shape[1], cv_image.shape[0]), PILImage.NEAREST
            )
            mask_resized = np.array(mask_resized, dtype=bool)

            # Create a binary mask (white for non-masked regions, black for masked regions)
            binary_mask = np.ones(cv_image.shape[:2], dtype=np.uint8) * 255  # Initialize as white
            binary_mask[mask_resized] = 0  # Set masked areas to black

            # Apply the mask to the original image to create the segmented image
            segmented_image = cv_image.copy()
            segmented_image[mask_resized] = [0, 0, 0]  # Make masked areas black

            return segmented_image, binary_mask

        except Exception as e:
            rospy.logerr(f"Error processing masks: {e}")
            # Return original image and a blank binary mask in case of an error
            blank_mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
            return cv_image, blank_mask
        
    def resize_image(self, image, scale_value=0.5):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]  # Get the height and width from the image
            new_w, new_h = int(w * scale_value), int(h * scale_value)

            # Perform resizing
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Log the resized image dimensions for verification
            rospy.loginfo(f"Resized image shape: {resized_image.shape}")
            
            return resized_image
        else:
            rospy.logerr("Input image is not a valid NumPy array.")
            return image  # Return original image if the type is not correct


    def publish_segmented_images(self, segmented_image,mask_image):
        """Publish the segmented image as a ROS message"""
        try:
            # Convert the segmented image (OpenCV format) to a ROS Image message
            ros_image_seg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")
            mask_image_seg = self.bridge.cv2_to_imgmsg(mask_image, encoding="mono8")
            
            # Publish the image
            self.image_pub_seg_image.publish(ros_image_seg)
            self.image_pub_seg_mask.publish(mask_image_seg)

            rospy.loginfo("Published segmented image.")
        except Exception as e:
            rospy.logerr(f"Error publishing image: {e}")

    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    segmenter = FastSAMSegmenter()
    segmenter.run()

