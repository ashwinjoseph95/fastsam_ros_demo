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
import random
from scipy.spatial import distance
import colorsys



class FastSAMSegmenter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('fastsam_segmenter_for_everything', anonymous=True)
        
        # Setup the ROS subscriber for compressed images
        self.image_sub = rospy.Subscriber('/zed2i/zed_node/left/image_rect_color/compressed', CompressedImage, self.image_callback)
        
        # Setup the ROS publisher for segmented images
        self.image_pub_seg_image = rospy.Publisher('/fastsam/segmented_image', Image, queue_size=10)
        
        # Initialize the bridge and FastSAM model
        self.bridge = CvBridge()
        self.model = FastSAM('/home/knadmin/Ashwin/vision_language_model_study/FastSAM/weights/FastSAM.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_count = 0  # Initialize frame_count
        self.object_centers = []
        self.object_colors = []
        self.color_palette = self.generate_distinct_colors(100)  # Generate 100 distinct colors
        self.color_index = 0

        rospy.loginfo("FastSAM Node Initialized.")

    def generate_distinct_colors(self,num_colors):
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            saturation = 0.9
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(tuple(int(x * 255) for x in rgb))
        return colors


    def image_callback(self, msg):
        try:
            # Skip frames (e.g., process every 10th frame)
            self.frame_count += 1
            if self.frame_count % 10 != 0:
                return  # Skip this frame

            start_time = time.time()
            # Convert the compressed image to a CV image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Perform segmentation using FastSAM
            everything_results = self.model(
                pil_image,
                device=self.device,
                retina_masks=True,
                imgsz=640,  # Reduce from 1024
                conf=0.3,   # Reduce from 0.4
                iou=0.7,    # Reduce from 0.9
            )

            # Initialize the FastSAMPrompt processor
            prompt_process = FastSAMPrompt(pil_image, everything_results, device=self.device)

            # Get everything prompt
            ann = prompt_process.everything_prompt()

            # Convert PIL image to numpy array
            original_rgb_array = np.array(pil_image)

            # Create a blank mask to store all segmentations
            segmentation_mask = np.zeros(original_rgb_array.shape, dtype=np.uint8)

            new_centers = []
            for i, mask in enumerate(ann):
                mask_np = mask.cpu().numpy().astype(bool)
                center = np.mean(np.argwhere(mask_np), axis=0)
                new_centers.append(center)

                # Find the closest existing object center
                if self.object_centers:
                    distances = [distance.euclidean(center, c) for c in self.object_centers]
                    min_distance = min(distances)
                    if min_distance < 50:  # Threshold for considering it the same object
                        color = self.object_colors[distances.index(min_distance)]
                    else:
                        color = self.color_palette[self.color_index % len(self.color_palette)]
                        self.color_index += 1
                else:
                    color = self.color_palette[self.color_index % len(self.color_palette)]
                    self.color_index += 1

                segmentation_mask[mask_np] = color

            # Update object centers and colors
            self.object_centers = new_centers
            self.object_colors = [segmentation_mask[tuple(map(int, center))] for center in new_centers]

            # Apply colormap to the segmentation mask
            colored_segmentation = segmentation_mask

            # Blend the original image with the colored segmentation
            alpha = 0.7  # Adjust this value to change the blend ratio
            blended_image = cv2.addWeighted(cv2.cvtColor(original_rgb_array, cv2.COLOR_RGB2BGR), 1 - alpha, colored_segmentation, alpha, 0)

            end_time = time.time()
            time_taken = end_time - start_time
            rospy.loginfo(f"Inference time per image: {time_taken:.3f} seconds")

            # Publish the segmented image
            self.publish_segmented_images(blended_image)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def publish_segmented_images(self, segmented_image):
        """Publish the segmented image as a ROS message"""
        try:
            # Convert the segmented image (OpenCV format) to a ROS Image message
            ros_image_seg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")
            
            # Publish the image
            self.image_pub_seg_image.publish(ros_image_seg)

            rospy.loginfo("Published segmented image.")
        except Exception as e:
            rospy.logerr(f"Error publishing image: {e}")

    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    segmenter = FastSAMSegmenter()
    segmenter.run()