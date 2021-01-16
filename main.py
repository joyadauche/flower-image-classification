import tensorflow as tf
import pathlib
import os
from shutil import copyfile
import random
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = keras.utils.get_file('flower_photos', origin=_URL, untar=True)
data_dir = pathlib.Path(data_dir)
try:
    main_dir = "./tmp/flower_photos/"

    train_dir = os.path.join(main_dir, "training")
    validation_dir = os.path.join(main_dir, "validation")

    daisy_train = os.path.join(train_dir, "daisy")
    dandelion_train = os.path.join(train_dir, "dandelion")
    roses_train = os.path.join(train_dir, "roses")
    sunflowers_train = os.path.join(train_dir, "sunflowers")
    tulips_train = os.path.join(train_dir, "tulips")

    daisy_test = os.path.join(validation_dir, "daisy")
    dandelion_test = os.path.join(validation_dir, "dandelion")
    roses_test = os.path.join(validation_dir, "roses")
    sunflowers_test = os.path.join(validation_dir, "sunflowers")
    tulips_test = os.path.join(validation_dir, "tulips")

    os.mkdir(main_dir)

    os.mkdir(train_dir)
    os.mkdir(validation_dir)

    os.mkdir(daisy_train)
    os.mkdir(dandelion_train)
    os.mkdir(roses_train)
    os.mkdir(sunflowers_train)
    os.mkdir(tulips_train)

    os.mkdir(daisy_test)
    os.mkdir(dandelion_test)
    os.mkdir(roses_test)
    os.mkdir(sunflowers_test)
    os.mkdir(tulips_test)
except OSError:
    pass


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.95:
            print('\n Reached 99% accuracy, so cancelling training')
            self.model.stop_training = True


def split_data(source, training, validation, split):
    data = os.listdir(source)
    data = random.sample(data, len(data))
    for count, file in enumerate(data):
        if (count < split * len(data)) and os.path.getsize(f"{source}/{file}") != 0:
            copyfile(f"{source}/{file}", f"{training}/{file}")
        elif (count >= split * len(data)) and os.path.getsize(f"{source}/{file}") != 0:
            copyfile(f"{source}/{file}", f"{validation}/{file}")


DAISY_SOURCE_DIR = os.path.join(data_dir, 'daisy')
TRAINING_DAISY_DIR = "./tmp/flower_photos/training/daisy/"
VALIDATION_DAISY_DIR = "./tmp/flower_photos/validation/daisy/"

DANDELION_SOURCE_DIR = os.path.join(data_dir, 'dandelion')
TRAINING_DANDELION_DIR = "./tmp/flower_photos/training/dandelion/"
VALIDATION_DANDELION_DIR = "./tmp/flower_photos/validation/dandelion/"

ROSES_SOURCE_DIR = os.path.join(data_dir, 'roses')
TRAINING_ROSES_DIR = "./tmp/flower_photos/training/roses/"
VALIDATION_ROSES_DIR = "./tmp/flower_photos/validation/roses/"

SUNFLOWER_SOURCE_DIR = os.path.join(data_dir, 'sunflowers')
TRAINING_SUNFLOWER_DIR = "./tmp/flower_photos/training/sunflowers/"
VALIDATION_SUNFLOWER_DIR = "./tmp/flower_photos/validation/sunflowers/"

TULIPS_SOURCE_DIR = os.path.join(data_dir, 'tulips')
TRAINING_TULIPS_DIR = "./tmp/flower_photos/training/tulips/"
VALIDATION_TULIPS_DIR = "./tmp/flower_photos/validation/tulips/"

split_size = .9
split_data(DAISY_SOURCE_DIR, TRAINING_DAISY_DIR, VALIDATION_DAISY_DIR, split_size)
split_data(DANDELION_SOURCE_DIR, TRAINING_DANDELION_DIR, VALIDATION_DANDELION_DIR, split_size)
split_data(ROSES_SOURCE_DIR, TRAINING_ROSES_DIR, VALIDATION_ROSES_DIR, split_size)
split_data(SUNFLOWER_SOURCE_DIR, TRAINING_SUNFLOWER_DIR, VALIDATION_SUNFLOWER_DIR, split_size)
split_data(TULIPS_SOURCE_DIR, TRAINING_TULIPS_DIR, VALIDATION_TULIPS_DIR, split_size)

print(len(os.listdir('./tmp/flower_photos/training/daisy/')))
print(len(os.listdir('./tmp/flower_photos/training/dandelion/')))
print(len(os.listdir('./tmp/flower_photos/training/roses/')))
print(len(os.listdir('./tmp/flower_photos/training/sunflowers/')))
print(len(os.listdir('./tmp/flower_photos/training/tulips/')))
print()
print(len(os.listdir('./tmp/flower_photos/validation/daisy/')))
print(len(os.listdir('./tmp/flower_photos/validation/dandelion/')))
print(len(os.listdir('./tmp/flower_photos/validation/roses/')))
print(len(os.listdir('./tmp/flower_photos/validation/sunflowers/')))
print(len(os.listdir('./tmp/flower_photos/validation/tulips/')))

callbacks = MyCallback()

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.
)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=20,
                                                              class_mode='categorical',
                                                              target_size=(150, 150))

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights('./tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

for layer in pre_trained_model.layers:
    layer.trainable = False
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
x = Flatten()(last_output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(5, activation='softmax')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_steps=108,
    callbacks=[callbacks]
)
model.save("flower_classifier.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
