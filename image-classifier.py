#Do not import any additional modules
import numpy as np
import cv2
import PIL
from PIL.Image import open
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import LinearSVC

# 3.1
Path = "./Project2_data/TrainingDataset/"
filelist = os.listdir(Path)

trainingSetDescriptors = np.empty((0,128))
numDescriptorsTrainingImages = []


for i in filelist:
    if i.endswith(".jpg"):
        I = np.array(open(Path + i).convert('L'))
        sift = cv2.SIFT_create()
        [f, descriptors] = sift.detectAndCompute(I, None)
        numDescriptors, dimension = descriptors.shape
        trainingSetDescriptors = np.vstack((trainingSetDescriptors, descriptors))
        numDescriptorsTrainingImages.append(numDescriptors)


#3.2
kmns = KMeans(n_clusters=100, n_init=2, max_iter=100, random_state=0).fit(trainingSetDescriptors)


#3.3
totalCount = 0
histogramTrainingImagesList = []
trainingDataHistogram = np.empty((0,100))

for num in numDescriptorsTrainingImages:
    oldCount = totalCount
    totalCount = totalCount + num
    arr = kmns.labels_[oldCount:totalCount]
    hist, bins = np.histogram(arr, bins=100, range=(0,100), density=True )
    histogramTrainingImagesList.append(hist)
    trainingDataHistogram = np.vstack((trainingDataHistogram, hist))


#3.4

Path = "./Project2_data/TestingDataset/"
filelist = os.listdir(Path)
testdataset = np.empty((0,128))
numFeaturesPerImageTestSet = []

for i in filelist:
    if i.endswith(".jpg"):  # You could also add "and i.startswith('f')

        I = np.array(open(Path + i).convert('L'))
        sift = cv2.SIFT_create()
        [f, d] = sift.detectAndCompute(I, None)
        a, b = d.shape
        testdataset = np.vstack((testdataset, d))

    numFeaturesPerImageTestSet.append(a)


testBinsArr = kmns.predict(testdataset)
totalCount = 0
histogramTestDataArr = []
testDataHistogram = np.empty((0,100))

for num in numFeaturesPerImageTestSet :
    oldCount = totalCount
    totalCount = totalCount + num
    arr = testBinsArr[oldCount:totalCount]
    hist, bins = np.histogram(arr, bins=100, range=(0,100), density=True )
    histogramTestDataArr.append(hist)
    testDataHistogram = np.vstack((testDataHistogram, hist))



# 3.5 KNN classification
imageType = []
for i in range(157):
    if i < 50:
        imageType.append(0)
    elif i < 99:
        imageType.append(1)
    else:
        imageType.append(2)



neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(trainingDataHistogram, imageType)

print("\n")
print("Actual Image Classification: 0 = Butterfly, 1 = Hat, 2 = Airplane")
print("[0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]")
print("")

print("K Nearest Neighbors")
print(neigh.predict(testDataHistogram))
print("")

#3.6 Linear
clf = LinearSVC()
clf.fit(trainingDataHistogram, imageType)
print("Linear SVM")
print(clf.predict(testDataHistogram))
print("")

#3.7 RBF Kernel
clf = svm.SVC()
clf.fit(trainingDataHistogram, imageType)
print("RBF Kernel SVM")
print(clf.predict(testDataHistogram))

