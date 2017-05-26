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
for bus_stop in range(1,9):
    for train_images in range(90):
        y_train.append(bus_stop)
y_train = np.array(y_train)

print y_train.shape

def get_one_image(class_name, rand_num):
    rand_num=rand_num[0]
    if class_name == 1:
        return image_resize(plt.imread("./TestData/7H/Test" + str(rand_num) + ".jpg")).flatten()
    if class_name == 2:
        return image_resize(plt.imread("./TestData/AN West Depot/Test" + str(rand_num) + ".jpg")).flatten()
    if class_name == 3:
        return image_resize(plt.imread("./TestData/Coll/Test" + str(rand_num) + ".jpg")).flatten()
    if class_name == 4:
        return image_resize(plt.imread("./TestData/Mogs/Test" + str(rand_num) + ".jpg")).flatten()
    if class_name == 5:
        return image_resize(plt.imread("./TestData/Thiru/Test" + str(rand_num) + ".jpg")).flatten()
    if class_name == 6:
        return image_resize(plt.imread("./TestData/Gurunath/Test" + str(rand_num) + ".jpg")).flatten()
    if class_name == 7:
        return image_resize(plt.imread("./TestData/Incubation Centre/Test" + str(rand_num) + ".jpg")).flatten()
    if class_name == 8:
        return image_resize(plt.imread("./TestData/SOMCA/Test" + str(rand_num) + ".jpg")).flatten()
		
def compute_distances(X_train, X):
    X = np.array(X)
    X = X.astype("float32")
    X /= 255.0
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.sqrt((np.square(X).sum(axis=1, keepdims=True)) - (2*X.dot(X_train.T)) + (np.square(X_train).sum(axis=1)))
    return dists

for n_images in [2]:
    t1 = time()
    for iterx in range(100):
        print "Iter : " + str(iterx)
        class_name = np.random.randint(1, 9)
        print "Class name : " + str(class_name), "No. of Images : " + str(n_images)
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

        for k in range(1, 2):
            count = 0
            correct_classes = []
            for i in range(dis.shape[0]):
                l = y_train[np.argsort(dis[i, :])].flatten()
                closest_y = l[:k]
                correct_classes.append(Counter(closest_y).most_common(1)[0][0])
            correct_classes = np.array(correct_classes)
            print "Closest 10 images : " + str(l[:10])
            print (correct_classes)

        for var in range(1, 8):
            print "Accuracy : " + str(Counter(correct_classes).most_common(1)[0][1] / float(len(correct_classes)))
            if Counter(correct_classes).most_common(1)[0][1] / float(len(correct_classes)) > 0.5:
                print "Predicted as ",
                print correct_classes,
                print "Groundtruth : " + str(class_name)
                break
            else:
                print "Fetching New Image ",
                new_image_no = np.random.randint(1, 31, size=1)
                while new_image_no in test_images:
                    new_image_no = np.random.randint(1, 31, size=1)
                print test_images, new_image_no
                list(test_images).append(new_image_no)
                new_test = get_one_image(class_name=class_name, rand_num=new_image_no)
                X_test = np.vstack((X_test, new_test))
                print (X_test.shape)
                dis_new = compute_distances(X_train, X_test)
                correct_classes1 = []
                for ii in range(dis_new.shape[0]):
                    l1 = y_train[np.argsort(dis_new[ii, :])].flatten()
                    closest_y1 = l1[:k]
                    correct_classes1.append(Counter(closest_y1).most_common(1)[0][0])
                if Counter(correct_classes1).most_common(1)[0][1] / float(len(correct_classes1)) < 0.5:
                    print "Back again " + str(Counter(correct_classes1).most_common(1)[0][1] / float(len(correct_classes1)))
                    print "Predicted as : ",
                    print correct_classes1,
                    print "Ground Truth : " + str(class_name)
                else:
                    print "Predicted as : ",
                    print correct_classes1,
                    print "Ground Truth : " + str(class_name)
                    print "Accuracy now is : " + str(Counter(correct_classes1).most_common(1)[0][1] / float(len(correct_classes1)))
                    break
        print
    t2 = time()
    print (t2-t1)
