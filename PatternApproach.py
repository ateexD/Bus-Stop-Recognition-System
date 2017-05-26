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
    X_train=np.vstack((X_train,image_resize(img).flatten().astype("float32")))
print len(X_train)

X_train = X_train / 255.0

y_train = []
for bus_stop in range(1, 9):
    for train_images in range(90):
        y_train.append(bus_stop)
y_train = np.array(y_train)

def compute_distances(X_train, X):
    X = np.array(X)
    X = X.astype("float32")
    X /= 255.0
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.sqrt((np.square(X).sum(axis=1, keepdims=True)) - (2*X.dot(X_train.T)) + (np.square(X_train).sum(axis=1)))
    return dists

for n_images in [2, 3, 5, 7, 9, 11]:
    t1 = time()
    for iterx in range(100):
        print "Iter : " + str(iterx)
        class_name = np.random.randint(1, 9)
        print class_name, n_images
        X_test = np.zeros((n_images, np.prod(X_train.shape[1:])))
        x = 0
        if class_name == 1:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestData/7H/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 2:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestData/AN West Depot/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 3:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestData/Coll/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 4:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestData/Mogs/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 5:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestData/Thiru/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 6:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestData/Gurunath/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 7:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestData/Incubation Centre/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
        elif class_name == 8:
            test_images = np.random.randint(1, 31, size=n_images)
            for i in test_images:
                X_test_temp = image_resize(plt.imread("./TestData/SOMCA/Test" + str(i) + ".jpg")).flatten()
                X_test[x, :] = X_test_temp
                x += 1
				
        dis = compute_distances(X_train, X_test)
		
        for k in range(1, 6):
            count = 0
            correct_classes = []
            for i in range(dis.shape[0]):
                l = y_train[np.argsort(dis[i, :])].flatten()
                closest_y = l[:k]
                correct_classes.append(Counter(closest_y).most_common(1)[0][0])
            correct_classes = np.array(correct_classes)
            print l[:10]
            for v in range(correct_classes.shape[0]):
                if correct_classes[v] == class_name:
                    count += 1
            print correct_classes
            accuracy = float(count) / dis.shape[0]
            print "K = " + str(k) + "; Accuracy : " + str(accuracy)
        print
    t2 = time()
    print t2-t1
