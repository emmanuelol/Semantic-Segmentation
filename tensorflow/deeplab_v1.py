import cv2
import tensorflow as tf
import numpy as np

graph_filename='/app/models/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb'


class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
 
    def __init__(self, model_filepath):
        # Initialize the model
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)
        #print(self.sess.graph.get_operations())

    def run(self,image):
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image=cv2.resize(image,(self.INPUT_SIZE,self.INPUT_SIZE))
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

##### color for the segmentation

def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
       A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

        Args:
            label: A 2D array with integer type, storing the segmentation label.

        Returns:
            result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

        Raises:
            ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

MODEL = DeepLabModel(graph_filename)

# webcam
# Create a VideoCapture object
cap = cv2.VideoCapture("/dev/video0")
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while(True):
  ret, frame = cap.read()
  if ret == True: 
      resized_im, seg_map = MODEL.run(frame)
      seg_image = label_to_color_image(seg_map).astype(np.uint8)
      res=cv2.resize(seg_image,(frame_width,frame_height),interpolation=cv2.INTER_CUBIC)
      
      #print(np.shape(seg_image))
      dst = cv2.addWeighted(frame,0.7,res,0.3,0)
      retval,mask_img = cv2.threshold(res, 0,1,cv2.THRESH_BINARY)
      img=frame*mask_img
      cv2.imshow('frame',dst)
      cv2.imshow('frame2',img)     
      # Press Q on keyboard to stop recording
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
 

 
# When everything done, release the video capture and video write objects
cap.release()

 
# Closes all the frames
cv2.destroyAllWindows()       