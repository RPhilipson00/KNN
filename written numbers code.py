#loads up all the packages, for some reason the solution didn't work with neighbors being imported here, so it is inside the sub now
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics 
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#assigns the dataset to a variable
digits = datasets.load_digits()
#menu subroutine to allow the user to choose what they want to do
def menu():
    choice = input("do you wish to view information about the data (0), run the learning algorithm (1), or exit (9)")
    if choice == "0":
        output()
    elif choice == "1":
        learn()
    elif choice == "9":
        exit()
    else:
        print("invalid entry")
        menu()
def output():
    classes = digits.target_names
    classesnum = len(classes)
    #outputs information about the dataset
    print("there are", classesnum, "classes: ", classes )
    print("the dataset has ", digits.data.shape[0], " instances, each with ", digits.data.shape[1], "features") 
    #rounds data to nearest 10 to estmiate how many times each number appears in the data
    dataperclass = (digits.data.shape[0])/classesnum
    roundeddata = round(dataperclass/10)*10
    digits.keys()
    digits.target = digits.target.astype(np.int8)
    print("each class has approximately",roundeddata , "samples")
    print("the data are all integers ranging from 0 to 16")
    menu()

def learn():
    
    _, axes = plt.subplots(2, 4)
    #sorts out the display of the learning results
    images_and_labels = list(zip(digits.images, digits.target))
    for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)

    #establishes number of samples
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    #assigns learning algorithm to 
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=3)

    #train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)
    classifier.fit(X_train, y_train)

    #predicted number for xtest
    predicted = classifier.predict(X_test)

    #puts data into table
    images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
    for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Prediction: %i' % prediction)

    #more output, showing prediction matrix
    #this matrix will show the confidence of the algorithm for each value
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))
    disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("confusion matrix:\n%s" % disp.confusion_matrix)

    plt.show()
    menu()

menu()
