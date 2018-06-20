import  glob
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

Image_width, Image_height = 299, 299
Training_Epochs = 2

Batch_Size = 32
Number_FC_Neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'

num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)

num_epoch = Training_Epochs
batch_size = Batch_Size

train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

train_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(Image_width, Image_height),
    batch_size=batch_size,
    seed=42
)

validation_generator = test_image_gen.flow_from_directory(
    validate_dir,
    target_size=(Image_width, Image_height),
    batch_size=batch_size,
    seed=42
)

InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)
print("Inception V3 base model without the last FC loaded")

x = InceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_FC_Neurons, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)


model = Model(inputs= InceptionV3_base_model.input, outputs = predictions)

model.summary()

print('\nPerforming Transfer Learning')

for layer in InceptionV3_base_model.layers:
    layer.trainable = False

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_transfer_learning = model.fit_generator(
    train_generator,
    epochs=num_epoch,
    steps_per_epoch=num_train_samples//batch_size,
    validation_data=validation_generator,
    validation_steps=num_validate_samples//batch_size,
    class_weight='auto'
)

model.save('inceptionv3-transfer-learning.model')