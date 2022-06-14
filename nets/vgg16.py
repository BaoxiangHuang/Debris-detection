
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers,Input,models

def VGG16(inputs):

  # Block 1
    x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = layers.Conv2D(
      64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
      128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
      256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    feat1=x
    # Block 4
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    feat2=x
    # Block 5
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
      512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # model = models.Model(inputs=inputs, outputs=x)
    # model.load_weights('vgg16_imagenet.h5')
    # model.summary()
    feat3=x
    return feat1,feat2,feat3
#   return model
#
# VGG16(inputs=Input([416,416,3]))
