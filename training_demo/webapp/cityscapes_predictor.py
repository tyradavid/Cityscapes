# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 21:21:22 2020

@author: tyra1
"""

import io
import numpy as np
import glob
from IPython.display import display
from PIL import Image

from PIL import Image, ImageDraw, ImageFont


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

labelmap_path=r"annotations\label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)


tf.keras.backend.clear_session()
model = tf.saved_model.load(r"exported-models")


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


from io import BytesIO
import base64

class CITYSCAPES_predictor:
    def __init__(self):
        self.model = model

    def predict(self, request):
        """
        This method reads the file uploaded from the Flask application POST request,
        and performs a prediction using the CITYSCAPES model.
        """
        f = request.files['image']
        
        img = Image.open(f)
        
        image = img.convert('RGB')
        
        image_np = load_image_into_numpy_array(image)
        output_dict = run_inference_for_single_image(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=2, 
            min_score_thresh=0.45, 
            skip_scores=True)
    
        result_image = Image.fromarray(image_np)
        
        raw_bytes = BytesIO()
        result_image.save(raw_bytes, "PNG")
        
        return base64.b64encode(raw_bytes.getvalue()).decode("utf-8") 
        




