import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = r"PetImages"
CATEGORIES = ['Dog', 'Cat']

def create_training_data(DATADIR, CATEGORIES):  

    data = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:    
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (50, 50))

                data.append([new_array, class_num])

            except Exception as e:
                pass

    return data

#--------------------------------Creating and saving our Data---------------------------------------

# training_data = create_training_data(DATADIR, CATEGORIES)
# random.shuffle(training_data)

# X = []
# y = []

# for features, label in training_data:
#     X.append(features)
#     y.append(label)

# X = np.array(X).reshape(-1, 50, 50, 1)

# pickle_out = open('data/X.pickle', 'wb')
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open('data/y.pickle', 'wb')
# pickle.dump(y, pickle_out)
# pickle_out.close()

#------------------------------Loading data-----------------------------------
pickle_in = open('data/X.pickle', 'rb')
X = pickle.load(pickle_in)

pickle_in = open(data/'y.pickle', 'rb')
y = pickle.load(pickle_in)