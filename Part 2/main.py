import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from win10toast import ToastNotifier
import cv2
import matplotlib.pyplot as plt

# toast = ToastNotifier() # For notifying when training is done

#-----------------------Getting Training Data-----------------------------
# X = pickle.load(open('data/X.pickle', 'rb'))
# y = pickle.load(open('data/y.pickle', 'rb'))

# #Normalising
# X = np.array(X/255.0)
# y = np.array(y)

#-------------------------Defining the network-----------------------------
# model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape = X.shape[1:])) 
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(Dense(64))
# model.add(Activation('selu'))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))

#---------------------Training the model---------------------
# model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# model.fit(X, y, batch_size=32, validation_split=0.1, epochs=8)

# toast.show_toast("Model Training Done","The program has finished training the model",duration=5) # Send notification

# model.save('Part-2-Models/model1.model') #Saving model


#--------------------Loading model and using it to predict----------------
# Created simple implementation to predict on any image

model = tf.keras.models.load_model('Part-2-Models\model1.model')

while True:
    inp = input(">")

    if inp == "help":
        print("""
        >open - gives prediction on image given
        >open_show - does same thing as open but shows the image as well
        >exit - exits program
        >help - shows this message
        """)

    if inp == "exit":
        break

    if inp == "open":
        directory = input("Enter image directory - ")

        image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
        model_image = cv2.resize(image, (50, 50))
        model_image = np.array(model_image).reshape(-1, 50, 50, 1)

        predict = model.predict([model_image])

        if predict[0] >= 1:
            print("Cat\n")

        elif predict[0] <= 0:
            print("Dog\n")

        else:
            print(predict[0])

    if inp == "open_show":
        directory = input("Enter image directory - ")

        image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
        model_image = cv2.resize(image, (50, 50))
        model_image = np.array(model_image).reshape(-1, 50, 50, 1)

        predict = model.predict([model_image])

        if predict[0] >= 1:
            print("Cat\n")

        elif predict[0] <= 0:
            print("Dog\n")

        else:
            print(predict[0])

        plt.imshow(image)
        plt.show()