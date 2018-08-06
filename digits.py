import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import  svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

labelled_images = pd.read_csv("C:\\Users\\KudzanayiDarylMutamb\\Downloads\\train (1).csv")

# dataset has 28000 images
images = labelled_images.iloc[0:28000,1:]
labels = labelled_images.iloc[0:28000,:1]



scaler = preprocessing.StandardScaler()
images = scaler.fit_transform(images);
train_images , test_images, train_labels, test_labels = train_test_split(images,labels,train_size=0.8, random_state=0);


mlp = MLPClassifier(hidden_layer_sizes=(784, 784, 784))

mlp.fit(train_images,train_labels.values.ravel());

predictions = mlp.predict(test_images)
print(confusion_matrix(test_labels,predictions))
print(classification_report(test_labels,predictions));


test_data=pd.read_csv('C:\\Users\\KudzanayiDarylMutamb\\Downloads\\test (1).csv')
scaler.fit_transform(test_data)

results=mlp.predict(test_data)

df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)