import sys
import cv2
import threading
import tkinter as tk
import tkinter.ttk as ttk
from queue import Queue
import PIL.Image
from PIL import ImageTk
import tkinter.font as tkFont
from tkinter import *
from tkinter.ttk import *
from predictLetter import predictletter

class App(tk.Frame):
    
    def __init__(self, parent, title):
        tk.Frame.__init__(self, parent)
        self.is_running = False
        self.thread = None
        self.queue = Queue()
        self.progress = Progressbar(self, orient=HORIZONTAL, length=100, mode='indeterminate')
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (500, 500), "white"))
        parent.wm_withdraw()
        parent.wm_title(title)
        self.create_ui()
        self.grid(sticky=tk.NSEW)
        self.bind('<<MessageGenerated>>', self.on_next_frame)
        parent.wm_protocol("WM_DELETE_WINDOW", self.on_destroy)
        parent.grid_rowconfigure(0, weight = 1)
        parent.grid_columnconfigure(0, weight = 1)
        parent.wm_deiconify()
    
    

    def create_ui(self):
        self.button_frame = ttk.Frame(self)
        self.label1 = ttk.Label(self, text="Show label here",font=tkFont.Font(family="Helvetica", size=20))
        self.label1.pack()
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.RIGHT)
        self.start_button = ttk.Button(self.button_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.RIGHT)
        self.view = ttk.Label(self, image=self.photo)
        self.view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def on_destroy(self):
        self.stop()
        self.after(20)
        if self.thread is not None:
            self.thread.join(0.2)
        self.winfo_toplevel().destroy()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.is_running = False
        cap.release()
        cv2.destroyAllWindows()

    def videoLoop(self, mirror=False):
        #self.progress.grid(row=1,column=0)
        #self.progress.start()
        import os
        import tensorflow as tf
        from object_detection.utils import config_util
        import cv2 
        import numpy as np
        import uuid
        import time
        from PIL import Image
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        
        CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
        PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
        PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'
        
        paths = {
            'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
            'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
            'APIMODEL_PATH': os.path.join('Tensorflow','models'),
            'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
            'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
            'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
            'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
            'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
            'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
            'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
            'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
            'PROTOC_PATH':os.path.join('Tensorflow','protoc')
         }
        
        files = {
            'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
            'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
            'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
        }
        
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-4')).expect_partial()
        
        @tf.function
        def detect_fn(image):
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        global cap
        No=0
        cap = cv2.VideoCapture(No)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        #self.progress.stop()
        #self.progress.grid_forget()
        while self.is_running:
            ret, frame = cap.read()
            image_np = np.array(frame)            
                          
                      
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
        
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
        
            _,box=viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.8,
                        agnostic_mode=False)
            
            if  detections['detection_scores'][0] > 0.994:
                im_pil = Image.fromarray(image_np)
                im_width, im_height = im_pil.size    
                ymin=box[0][0]
                ymax=box[0][2]
                xmin=box[0][1]
                xmax=box[0][3]
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                croppedimg = im_pil.crop((left-10, top-10, right+10, bottom+6))
                image = np.asarray(croppedimg)
                imgname = "C:\\Users\\Dev1\\Desktop\\saved\\"+'saved'+'{}.jpg'.format(str(uuid.uuid1()))
                cv2.imwrite(imgname, image)
                text=predictletter(image)
                print(text)
                #do  needful processes here with  the text
                time.sleep(5)


            image = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
            self.queue.put(image)
            self.event_generate('<<MessageGenerated>>')

    def on_next_frame(self, eventargs):
        if not self.queue.empty():
            image = self.queue.get()
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image)
            self.view.configure(image=self.photo)


def main(args):
    root = tk.Tk()
    app = App(root, "ID detector")
    root.mainloop()

if __name__ == '__main__':
    sys.exit(main(sys.argv))