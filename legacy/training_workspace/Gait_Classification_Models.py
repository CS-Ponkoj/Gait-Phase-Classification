from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras import applications
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import device
from matplotlib import pyplot
from PIL import ImageFile
import sys
from keras_preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf