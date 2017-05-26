import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
from time import time
from collections import Counter

def image_resize(im):
    im = sc.imresize(im, (32, 32, 3))
    return im

X_train = np.array(image_resize(plt.imread("./TrainData/Train1.jpg")).flatten().astype("float32"))
for i in range(2, 721):
    if i % 25 == 0:
        print "Reading train image "+ str(i)
    img = plt.imread("./TrainData/Train" + str(i) + ".jpg")
    X_train = np.vstack((X_train,image_resize(img).flatten().astype("float32")))
print len(X_train)

X_train = X_train / 255.0

y_train = []
for bus_stop in range(1, 9):
    for train_images in range(90):
        y_train.append(bus_stop)
y_train = np.array(y_train)

y_test = []
for bus_stop in range(1, 9):
    for test_images in range(30):
        y_test.append(bus_stop)
y_test = np.array(y_test)

print (y_test.shape)
def compute_distances(X_train, X):
    X = np.array(X)
    X = X.astype("float32")
    X /= 255.0
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.sqrt((np.square(X).sum(axis=1, keepdims=True)) - (2*X.dot(X_train.T)) + (np.square(X_train).sum(axis=1)))
    return dists

x = 0
X_test = np.zeros((240, np.prod(X_train.shape[1:])))
for i in range(1, 31):
    X_test_temp = image_resize(plt.imread("./TestData/7H/Test" + str(i) + ".jpg")).flatten()
    X_test[x, :] = X_test_temp
    x += 1
for i in range(1, 31):
    X_test_temp = image_resize(plt.imread("./TestData/AN West Depot/Test" + str(i) + ".jpg")).flatten()
    X_test[x, :] = X_test_temp
    x += 1
for i in range(1, 31):
    X_test_temp = image_resize(plt.imread("./TestData/Coll/Test" + str(i) + ".jpg")).flatten()
    X_test[x, :] = X_test_temp
    x += 1
for i in range(1, 31):
    X_test_temp = image_resize(plt.imread("./TestData/Mogs/Test" + str(i) + ".jpg")).flatten()
    X_test[x, :] = X_test_temp
    x += 1
for i in range(1, 31):
    X_test_temp = image_resize(plt.imread("./TestData/Thiru/Test" + str(i) + ".jpg")).flatten()
    X_test[x, :] = X_test_temp
    x += 1
for i in range(1, 31):
    X_test_temp = image_resize(plt.imread("./TestData/Gurunath/Test" + str(i) + ".jpg")).flatten()
    X_test[x, :] = X_test_temp
    x += 1
for i in range(1, 31):
    X_test_temp = image_resize(plt.imread("./TestData/Incubation Centre/Test" + str(i) + ".jpg")).flatten()
    X_test[x, :] = X_test_temp
    x += 1
for i in range(1, 31):
    X_test_temp = image_resize(plt.imread("./TestData/SOMCA/Test" + str(i) + ".jpg")).flatten()
    X_test[x, :] = X_test_temp
    x += 1

print X_test.shape

dis = compute_distances(X_train,X_test)

for k in range(1, 6):
    t1 = time()
    count = 0
    correct_classes = []
    for i in range(dis.shape[0]):
        l = y_train[np.argsort(dis[i, :])].flatten()
        closest_y = l[:k]
        correct_classes.append(Counter(closest_y).most_common(1)[0][0])
    correct_classes = np.array(correct_classes)
    print "Closest 10 images : " + str(l[:10])
    correct_classes_final = np.sum([correct_classes == y_test])
    print "K = " + str(k) + "; Accuracy : " + str(float(correct_classes_final)/dis.shape[0])
    print
    t2 = time()
    print t2-t1
