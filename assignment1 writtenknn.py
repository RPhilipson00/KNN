import pandas as pd
import numpy as np
from sklearn import datasets
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
import operator
import random
digits = datasets.load_digits()

def euclidean_distance(img_a, img_b):
    #Finds the distance between 2 images: img_a, img_b
    return np.sqrt(np.sum((img_a-img_b)**2))

#allows the user to input the training set, then assigns data to variables
x=1
print("input training length 1-1797")
x = int(input())
X_train = digits.data[0:x]
Y_train = digits.target[0:x]

pred = 0

#outputs 10 random tests from the dataset
for n in range (1,11):
    pred = random.randint(0,1797)
    X_test = digits.data[pred]
    l = len(X_train)
    distance = np.zeros(l)
    #finds the lowest euclidean distance, meaning the most likely prediction
    for i in range(l):
        distance[i] = euclidean_distance(X_train[i],X_test)
    
    min_index = np.argmin(distance)
    print ("test ", n, "Preditcted: ", Y_train[min_index], "actual: ", digits.target[pred])

#calculates error over the whole dataset
l = len(X_train)
no_err = 0
distance = np.zeros(l)
for j in range(1697,1797):
 X_test = digits.data[j]
 for i in range(l):
  distance[i] = euclidean_distance(X_train[i],X_test)
 min_index = np.argmin(distance)
 #if the prediction isn't correct 1 error is added
 if Y_train[min_index] != digits.target[j]:
  no_err+=1
print ("Total errors over whole dataset = ",(no_err))






