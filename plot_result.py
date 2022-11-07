from infer import *
from utils import *
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

x_true = []
x_pred = []

y_true = []
y_pred = []

z_true = []
z_pred = []

dists = []
x_dis = []
y_dis = []
z_dis = []

xyz_pred = []

print('len', len(os.listdir(ROUTE)))

for file in os.listdir(ROUTE)[:]:
    filepath = ROUTE + '/' + file
    ang_vertical, ang_horizon, ang_rot = infer_face_pose.predict_angle_from_path(filepath)

    row = df[df['image']==file]

    try:
        list_true = np.array(row.values[0][1:])
        list_pred = np.array([ang_vertical, ang_horizon, ang_rot])

        x_true.append(list_true[0])
        x_pred.append(list_pred[0])

        y_true.append(list_true[1])
        y_pred.append(list_pred[1])

        z_true.append(list_true[2])
        z_pred.append(list_pred[2])

        dist_d = np.abs(list_true - list_pred)

        x_dis.append(dist_d[0])
        y_dis.append(dist_d[1])
        z_dis.append(dist_d[2])

        xyz_pred.append(list_pred)

        dists.append(dist_d)
    except:
        pass

x_true = np.array(x_true)
x_pred = np.array(x_pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

z_true = np.array(z_true)
z_pred = np.array(z_pred)

from sklearn.linear_model import LinearRegression

regression = LinearRegression()
x_pred = np.reshape(x_pred, (-1, 1))
regression.fit(x_pred, x_true)
print('X regression.coef_', regression.coef_)
print('X regression.intercept_', regression.intercept_)

# regression = LinearRegression()
# print(np.shape(xyz_pred))
# regression.fit(xyz_pred, x_true)
# print('X regression.coef_', regression.coef_)
# print('X regression.intercept_', regression.intercept_)

# regression = LinearRegression()
# regression.fit(vers, x_true)
# print('X regression.coef_', regression.coef_)
# print('X regression.intercept_', regression.intercept_)

plt.figure()
plt.scatter(x_true, x_pred)
# plt.scatter(x_true, regression.predict(vers))

plt.plot(regression.predict(x_pred), x_pred, color='red', linewidth=3)
# plt.plot(regression.predict(xyz_pred), x_pred, color='red', linewidth=3)
# plt.plot(regression.predict(vers), x_pred, color='red', linewidth=3)

plt.xlabel("X true")
plt.ylabel("X pred")
plt.title("Euclide X")
plt.show()
plt.savefig(f'x_plot.png')

regression = LinearRegression()
y_pred = np.reshape(y_pred, (-1, 1))
regression.fit(y_pred, y_true)
print('Y regression.coef_', regression.coef_)
print('Y regression.intercept_', regression.intercept_)

plt.figure()
plt.scatter(y_true, y_pred)
plt.plot(regression.predict(y_pred), y_pred, color='red', linewidth=3)
plt.xlabel("Y true")
plt.ylabel("Y pred")
plt.title("Euclide Y")
plt.show()
plt.savefig(f'y_plot.png')

regression = LinearRegression()
z_pred = np.reshape(z_pred, (-1, 1))
regression.fit(z_pred, z_true)
print('Z regression.coef_', regression.coef_)
print('Z regression.intercept_', regression.intercept_)

plt.figure()
plt.scatter(z_true, z_pred)
plt.plot(regression.predict(z_pred), z_pred, color='red', linewidth=3)
plt.xlabel("Z true")
plt.ylabel("Z pred")
plt.title("Euclide Z")
plt.show()
plt.savefig(f'z_plot.png')

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