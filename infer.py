import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import math

class InferFacePose:
    def __init__(self, im_size, weight):
        self.im_size = im_size
        self.img_size = (im_size, im_size)
        self.weight = weight
        self.model = self.get_model()
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ])
        self.focal_length = self.img_size[1]
        self.camera_center = (self.img_size[1] / 2, self.img_size[0] / 2)
        self.camera_matrix = np.array(
                    [[self.focal_length, 0, self.camera_center[0]],
                    [0, self.focal_length, self.camera_center[1]],
                    [0, 0, 1]], dtype="float")
        self.dist_coeffs = np.zeros((4, 1))

    def preprocess_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.image.resize(img, self.img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def draw_marks(self, image, marks, mark_size=3, color=(0, 255, 0), line_width=-1):
        """Draw the marks in image.
        Args:
            image: the image on which to be drawn.
            marks: points coordinates in a numpy array.
            mark_size: the size of the marks.
            color: the color of the marks, in BGR format, ranges 0~255.
            line_width: the width of the mark's outline. Set to -1 to fill it.
        """
        # We are drawing in an image, this is a 2D situation.
        image_copy = image.copy()
        for point in marks:
            cv2.circle(image_copy, (int(point[0]), int(point[1])),
                    mark_size, color, line_width, cv2.LINE_AA)
        return image_copy

    def get_model(self):
        """
        can change strategy to gpu, gpus
        strategy = tf.distribute.MirroredStrategy()
        strategy = tf.distribute.get_strategy()
        """
        with tf.device('/cpu:0'):
            model = tf.keras.models.load_model(self.weight)
            model.summary()
        return model

    def predict_marks(self, img):
        img_batch = tf.expand_dims(img, 0)
        pred = self.model.predict(img_batch, verbose=0)
        marks = np.array(pred[0]) * self.im_size
        """
        return array (68, 2)
            68 keypoints (x, y)
        """
        return marks

    def get_3d_box(self, img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
        point_3d = []
        dist_coeffs = np.zeros((4,1))
        rear_size = 1
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = img.shape[1]
        front_depth = front_size*2
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d img points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                        rotation_vector,
                                        translation_vector,
                                        camera_matrix,
                                        dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        
        k = (point_2d[5] + point_2d[8])//2
        
        return(point_2d[2], k)

    def predict_angle(self, img, get_view_point=False):
        """
        return ang_vertical, ang_horizon
        angle (degree)
        ang_vertical: vertical angle
        ang_horizon: horizon angle
        """
        marks = self.predict_marks(img)
        
        marks = marks[[30,      # Nose tip
                        8,      # Chin
                        36,     # Left eye left corner
                        45,     # Right eye right corner
                        48,     # Left Mouth corner
                        54]     # Right mouth corner
                    ]
      
        eye_center = ((marks[2][0] + marks[3][0]) / 2, (marks[2][1] + marks[3][1]) / 2)
        dx = abs(marks[2][0] - marks[3][0])
        dy = abs(marks[2][1] - marks[3][1])
        ang_rot = np.arctan2(dy, dx)
        ang_rot = ang_rot * 180 / math.pi  

        (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, 
                                                                marks, 
                                                                self.camera_matrix, 
                                                                self.dist_coeffs, 
                                                                flags=cv2.SOLVEPNP_UPNP)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), 
                                                         rotation_vector, 
                                                         translation_vector, 
                                                         self.camera_matrix, 
                                                         self.dist_coeffs)
        
        x1, x2 = self.get_3d_box(img, rotation_vector, translation_vector, self.camera_matrix)

        p1 = (int(marks[0][0]), int(marks[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang_vertical = int(math.degrees(math.atan(m)))
        except:
            ang_vertical = 90
            
        try:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            ang_horizon = int(math.degrees(math.atan(-1/m)))
        except:
            ang_horizon = 90

        ang_vertical /= -4
        ang_horizon /= 4

        if get_view_point:
            return ang_vertical, ang_horizon, ang_rot, p1, p2

        return ang_vertical, ang_horizon, ang_rot

    def predict_marks_from_path(self, path):
        img = self.preprocess_img(path)
        marks = self.predict_marks(img)
        return marks

    def predict_angle_from_path(self, path):
        img = self.preprocess_img(path)
        ang_vertical, ang_horizon, ang_rot = self.predict_angle(img)
        return ang_vertical, ang_horizon, ang_rot

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]=""

    im_size = 192

    weight = '/home/lap14880/hieunmt/facepose/tf_facepose_pipeline/best_model_facepose_EfficientNetV1B3_192_68.h5'

    infer_face_pose = InferFacePose(im_size, weight)

    # img_path = './sample/0.jpg'
    img_path = './othersample/nhut1.png'

    img = infer_face_pose.preprocess_img(img_path)
    marks = infer_face_pose.predict_marks(img)
    ang_vertical, ang_horizon, ang_rot = infer_face_pose.predict_angle(img)

    print('ang_vertical, ang_horizon, ang_rot:', ang_vertical, ang_horizon, ang_rot)

    infer_folder = 'infer'

    try:
        shutil.rmtree(infer_folder)
    except:
        pass

    try:
        os.mkdir(infer_folder)
    except:
        pass

    img = np.array(img) * 255
    img_write = infer_face_pose.draw_marks(img, marks)
    # cv2.line(img_write, p1, p2, (255,255,255), 2)
    
    cv2.imwrite(f"./{infer_folder}/ori.jpg", img)  
    cv2.imwrite(f"./{infer_folder}/infer.jpg", img_write) 




