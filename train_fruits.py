import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation , Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import os
from os import chdir



num_classes=120
img_rows=32
img_cols=32
batch_size=16

train_data_dir="./fruits-360/train";
valid_data_dir="./fruits-360/valid";




#Genrating image data
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest")

valid_datagen=ImageDataGenerator(
    rescale=1./255)


#Applying data generators

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows,img_cols),
    batch_size=16,
    class_mode="categorical",
    shuffle=True
    )

valid_generator=valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_rows,img_cols),
    batch_size=16,
    class_mode="categorical"
    )

#Our Net

model=Sequential();

#1st layer

model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 padding="same",
                 input_shape=(img_rows,img_cols,3)))

model.add(Activation("relu"))

#2nd layer
model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#3rd layer
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))
model.add(Activation("relu"));


#4th layer
model.add(Conv2D(filters=64,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


#5th layer

model.add(Flatten());
model.add(Dense(units=500));
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))  #this is output layer so use number of classes value
model.add(Activation("softmax"))

print(model.summary())

checkpoint=ModelCheckpoint(filepath="fruits_cnn_{val_accuracy:.2f}.h5",
                           monitor="val_loss",
                           save_best_only=True,
                           verbose=1)


earlystop=EarlyStopping(monitor="val_loss",
                min_delta=0,
                patience=3,
                verbose=1,
                restore_best_weights=True);

reduce_lr=ReduceLROnPlateau(monitor="val_loss",
                            factor=0.2,
                            patience=3,
                            verbose=1,
                            min_delta=0.0001)


callbacks=[earlystop,checkpoint]

model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(lr=0.001),
              metrics=["accuracy"]);

nb_train_samples=60498
nb_valid_samples=20622
epochs=5


history=model.fit_generator(train_generator,
                            steps_per_epoch=nb_train_samples//batch_size,
                            epochs=5,
                            callbacks=callbacks,
                            validation_data=valid_generator,
                            validation_steps=nb_valid_samples//batch_size)








print("working")
