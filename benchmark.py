from infer import *
from utils import *
import pandas as pd 

os.environ["CUDA_VISIBLE_DEVICES"]=""

set_memory_growth()

ROUTE = './unzip/helen_image'

im_size = 192

weight = '/home/lap14880/hieunmt/facepose/tf_facepose_pipeline/best_model_facepose_EfficientNetV1B3_192_68.h5'

infer_face_pose = InferFacePose(im_size, weight)

df = pd.read_csv('/home/lap14880/hieunmt/facepose/tf_facepose_pipeline/mlkit_facepose_test.csv')

dists = []
x_dis = []
y_dis = []
z_dis = []

for file in os.listdir(ROUTE):
    filepath = ROUTE + '/' + file
    ang_vertical, ang_horizon, ang_rot = infer_face_pose.predict_angle_from_path(filepath)

    row = df[df['image']==file]

    try:
        list_true = np.array(row.values[0][1:])
        list_pred = np.array([ang_vertical, ang_horizon, ang_rot])

        dist_d = np.abs(list_true - list_pred)

        x_dis.append(dist_d[0])
        y_dis.append(dist_d[1])
        z_dis.append(dist_d[2])

        dists.append(dist_d)
    except:
        pass

def describe(arrs, names):
    with open('benchmark.txt', 'w') as f:
        f.write('name,mean,median,min,max,range,variance,sd,per25,per50,per75\n')
        print(('name,mean,median,min,max,range,variance,sd,per25,per50,per75\n'))

        for i, arr in enumerate(arrs):
            _name = names[i]
            _mean = np.mean(arr)
            _median = np.median(arr)
            _min = np.amin(arr)
            _max = np.amax(arr)
            _range = np.ptp(arr)
            _variance = np.var(arr)
            _sd = np.std(arr)
            _per25 = np.percentile(arr, 25)
            _per50 = np.percentile(arr, 50)
            _per75 = np.percentile(arr, 75)

            f.write(f'{_name},{_mean},{_median},{_min},{_max},{_range},{_variance},{_sd},{_per25},{_per50},{_per75}\n')
            print(f'{_name},{_mean},{_median},{_min},{_max},{_range},{_variance},{_sd},{_per25},{_per50},{_per75}\n')
            
list_des = [x_dis, y_dis, z_dis, dists]
names = ['x', 'y', 'z', 'all']
describe(list_des, names)