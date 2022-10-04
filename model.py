from  keras.applications.inception_v3 import InceptionV3
from  keras.layers import Flatten, Dense, Dropout
from keras.models import Model
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

def plot_hist(hist):

    for metric in hist.history.keys() :
        if metric.startswith('val_'):
            break
        plt.plot(hist.history[metric])
        plt.plot(hist.history['val_'+metric])
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

data = pd.read_csv(os.path.join(os.getcwd(), 'train.csv'))

print(data.head())

data['label'] = data['label'].astype('str')


print("Label distribution ")
print(data["label"].value_counts())


train, test = train_test_split(data, test_size=0.2, shuffle=True)


print("Train Label distribution ")
print(train["label"].value_counts())


print("Test Label distribution ")
print(test["label"].value_counts())


data_generator = ImageDataGenerator(rescale=1./255)
train_gen = data_generator.flow_from_dataframe(train,
            directory=os.path.join(os.getcwd(), "train_data"),
            batch_size=32, class_mode='categorical', target_size=(150, 150),
            x_col="image_name", y_col="label")

test_gen = data_generator.flow_from_dataframe(test,
            directory=os.path.join(os.getcwd(), "train_data"),
            batch_size=32, class_mode='categorical', target_size=(150, 150),
            x_col="image_name", y_col="label")





num_classes = 6

# loading the model :
inception_model = InceptionV3(input_shape=(150, 150, 3),
                              include_top=False,
                              weights='imagenet')

# freezing the inception model layers from training
for layer in inception_model.layers:
    layer.trainable = False

# the personalized layers :
x = Flatten() (inception_model.output)
x = Dense(512, activation="relu") (x)
x = Dropout(0.5) (x)
x = Dense(num_classes, activation="softmax") (x)


model = Model(inception_model.input, x)

print(model.summary())

# creating checkpoint so that we save the best val acc model
checkpoint = ModelCheckpoint(filepath='./model.h5',save_best_only=True,
                             monitor='val_acc', mode="max")

model.compile("Adam", "categorical_crossentropy", metrics=["acc"])

history = model.fit(train_gen, epochs=10, verbose=2,
                              validation_data=test_gen, callbacks=[checkpoint])

print(history.history.keys())
print(history.history)

plot_hist(history)


model.evaluate(test_gen)


# model.save("model.h5")

