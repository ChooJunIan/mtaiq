import tensorflow as tf
import numpy as np
import pandas as pd
import os
import shutil
import wandb
import matplotlib.pyplot as plt
import scipy 

from wandb.keras import WandbMetricsLogger, WandbCallback

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.vis_utils import plot_model

from keras import backend as K

# load functions, classes
def plcc_tf(x, y):
    """PLCC metric"""
    xc = x - K.mean(x)
    yc = y - K.mean(y)
    return K.mean(xc*yc)/(K.std(x)*K.std(y) + K.epsilon())

def pearson_correlation(y_true, y_pred):
    # Subtract the mean from true and predicted values
    y_true_mean = K.mean(y_true)
    y_pred_mean = K.mean(y_pred)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    # Calculate covariance and standard deviation
    covariance = K.mean(y_true_centered * y_pred_centered)
    y_true_std = K.std(y_true)
    y_pred_std = K.std(y_pred)

    # Calculate Pearson correlation coefficient
    pearson_coefficient = covariance / (y_true_std * y_pred_std + K.epsilon())

    return pearson_coefficient

class CustomMetricCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Calculate the custom metric
        y_true = self.validation_data[1]
        y_pred = self.model.predict(self.validation_data[0])
        pearson_coefficient = pearson_correlation(y_true, y_pred)

        # Log the custom metric using wandb
        wandb.log({"val_pearson_coefficient": pearson_coefficient})

def combined_generator(gen1, gen2, gen3, gen4):
    while True:
        batch1 = next(gen1)
        batch2 = next(gen2)
        batch3 = next(gen3)
        batch4 = next(gen4)
        inputs = [batch1[0], batch2[0], batch3[0], batch4[0]]
        targets = [batch1[1], batch2[1], batch3[1], batch4[1]]
        yield inputs, targets

def plcc(x, y):
    '''Pearson Linear Correlation Coefficient'''
    x, y = np.float32(x), np.float32(y)
    return scipy.stats.pearsonr(x,y)[0]

def srocc(xs, ys):
    '''Spearman Rank Order Correlation Coefficient'''
    correlation, p_value = scipy.stats.spearmanr(xs, ys)
    return correlation

def rmse(y_test, y_pred):
    mse = np.mean((y_test - y_pred) ** 2)
    return np.sqrt(mse)

# Learning rate schedule function
def lr_schedule(epoch):
    if epoch < 10:
        return 0.0001
    elif epoch < 15:
        return 0.00001
    else:
        return 0.000001
    
# loading the datasets 
main_directory = '/media/workstation/BackupDrive/Dataset/'

# ava
ava_images = main_directory + 'data_512x/'
ava_multimodel_dataset = main_directory + 'multimodel_dataset/ava_multimodel_all_train_same_test.csv'
ava_df = pd.read_csv(ava_multimodel_dataset)
# ava_df.drop(columns='Unnamed: 0', inplace=True)

# para
para_images = main_directory + 'PARA_512x_resized/'
para_multimodel_dataset = main_directory + 'multimodel_dataset/para_multimodel_all_train.csv'
para_df = pd.read_csv(para_multimodel_dataset)
para_df.drop(columns='Unnamed: 0', inplace=True)

# koniq 
koniq_images = main_directory + 'koniq10k_512x_image_in_csv/'
koniq_multimodel_dataset = main_directory + 'multimodel_dataset/koniq_multimodel.csv'
koniq_df = pd.read_csv(koniq_multimodel_dataset)
koniq_df.drop(columns='Unnamed: 0', inplace=True)

# spaq
spaq_images = main_directory + 'SPAQ_512x_resized/'
spaq_multimodel_dataset = main_directory + 'multimodel_dataset/spaq_multimodel.csv'
spaq_df = pd.read_csv(spaq_multimodel_dataset)
spaq_df.drop(columns='Unnamed: 0', inplace=True)

# split into training validation and testing
ava_train_df = ava_df[ava_df['set']=='training']
ava_val_df = ava_df[ava_df['set']=='validation']
ava_test_df = ava_df[ava_df['set']=='test']

para_train_df = para_df[para_df['set']=='training']
para_val_df = para_df[para_df['set']=='validation']
para_test_df = para_df[para_df['set']=='test']

koniq_train_df = koniq_df[koniq_df['set']=='training']
koniq_val_df = koniq_df[koniq_df['set']=='validation']
koniq_test_df = koniq_df[koniq_df['set']=='test']

spaq_train_df = spaq_df[spaq_df['set']=='training']
spaq_val_df = spaq_df[spaq_df['set']=='validation']# breakpoint()
spaq_test_df = spaq_df[spaq_df['set']=='test']


# Set the path to image directories, and prepare image data generators for each dataset.
ava_images = main_directory + 'AVA/data_512x/'
para_images = main_directory + 'PARA/PARA_512x_resized/'
koniq_images = main_directory + 'koniq10k/koniq10k_512x_image_in_csv/'
spaq_images = main_directory + 'SPAQ dataset-20230407T121509Z-008/SPAQ_512x_resized/'

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ava
ava_train_generator = train_datagen.flow_from_dataframe(
    dataframe=ava_train_df,
    directory=ava_images, 
    x_col="ID", 
    y_col="scaled_MOS_aesthetic", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)

ava_val_generator = val_datagen.flow_from_dataframe(
    dataframe=ava_val_df, 
    directory=ava_images, 
    x_col="ID", 
    y_col="scaled_MOS_aesthetic", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)

ava_test_generator = test_datagen.flow_from_dataframe(
    dataframe=ava_test_df, 
    directory=ava_images, 
    x_col="ID", 
    y_col="scaled_MOS_aesthetic", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)
print('AVA generators complete\n')

# para
para_train_generator = train_datagen.flow_from_dataframe(
    dataframe=para_train_df,
    directory=para_images, 
    x_col="sessionId_imageName", 
    y_col="scaled_MOS_aesthetic", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)

para_val_generator = val_datagen.flow_from_dataframe(
    dataframe=para_val_df, 
    directory=para_images, 
    x_col="sessionId_imageName", 
    y_col="scaled_MOS_aesthetic", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)

para_test_generator = test_datagen.flow_from_dataframe(
    dataframe=para_test_df, 
    directory=para_images, 
    x_col="sessionId_imageName", 
    y_col="scaled_MOS_aesthetic", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)
print('PARA generators complete\n')

# koniq
koniq_train_generator = train_datagen.flow_from_dataframe(
    dataframe=koniq_train_df,
    directory=koniq_images, 
    x_col="image_name", 
    y_col="scaled_MOS_quality", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)

koniq_val_generator = val_datagen.flow_from_dataframe(
    dataframe=koniq_val_df, 
    directory=koniq_images, 
    x_col="image_name", 
    y_col="scaled_MOS_quality", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)

koniq_test_generator = test_datagen.flow_from_dataframe(
    dataframe=koniq_test_df, 
    directory=koniq_images, 
    x_col="image_name", 
    y_col="scaled_MOS_quality", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)
print('KoNIQ generators complete\n')

# spaq
spaq_train_generator = train_datagen.flow_from_dataframe(
    dataframe=spaq_train_df,
    directory=spaq_images, 
    x_col="Image name", 
    y_col="scaled_MOS_quality", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)

spaq_val_generator = val_datagen.flow_from_dataframe(
    dataframe=spaq_val_df, 
    directory=spaq_images, 
    x_col="Image name", 
    y_col="scaled_MOS_quality", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)

spaq_test_generator = test_datagen.flow_from_dataframe(
    dataframe=spaq_test_df, 
    directory=spaq_images, 
    x_col="Image name", 
    y_col="scaled_MOS_quality", 
    class_mode="raw", 
    target_size=(224, 224), 
    batch_size=16
)
print('SPAQ generators complete\n')

# initialise tracking images
wandb.init(
    # set the wandb project where this run will be logged
    project="multimodel_irnv2",
    name='multimodel_full_train',
    dir = "/media/workstation/BackupDrive/wandb_files/logs",

    # track hyperparameters and run metadata with wandb.config
    config={
        "fc1" : 2048,
        "activation1" : 'relu',
        "dropout1": 0.25,
        "fc2" : 1024,
        "activation2" : 'relu',
        "dropout2": 0.25,
        "fc3" : 256,
        "activation3" : 'relu',
        "dropout3": 0.5,
        "fc4" : 1,
        "activation4" : 'linear',
        "dropout4": 0,
        "learning_rate" : 0.0001,
        "optimizer": "adam",
        "loss": "mean_squared_error",
        "metric": "root_mean_squared_error",
        "epoch": 20,
        "batch_size": 16,
        "metric2" : "val_loss",
        "early_patience" : 10,
        "early_mode" : 'min',
        "early_min_delta" : 0.001,
        "plateau_patience" : 5,
        "plateau_mode" : "min",
        "plateau_factor" : 0.1,
        "plateau_min_lr" : 0.000001,
        "plateau_min_delta" : 0.001
    }
)

config = wandb.config

# build the model 
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# function to create outputs
def configure_layers(input, name='Output normal'):
    x = GlobalAveragePooling2D()(input)
    x = Dense(config.fc1, activation=config.activation1)(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout1)(x)

    x = Dense(config.fc2, activation=config.activation2)(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout2)(x)

    x = Dense(config.fc3, activation=config.activation3)(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout3)(x)

    predictions = Dense(config.fc4, activation=config.activation4, name=name)(x)
    return predictions

# Create the input layers
input_ava = Input(shape=(224, 224, 3), name='Input_AVA')
input_para = Input(shape=(224, 224, 3), name='Input_PARA')
input_koniq = Input(shape=(224, 224, 3), name='Input_KonIQ')
input_spaq = Input(shape=(224, 224, 3), name='Input_SPAQ')

# Create the output layers, include the input layers 
output_ava = configure_layers(base_model(input_ava), name='Output_AVA_aesthetic')
output_para = configure_layers(base_model(input_para), name='Output_PARA_aesthetic')
output_koniq = configure_layers(base_model(input_koniq), name='Output_KonIQ_quality')
output_spaq = configure_layers(base_model(input_spaq), name='Output_SPAQ_quality')

# Multi-input, multi-output model
model = Model(inputs=[input_ava, input_para, input_koniq, input_spaq], 
              outputs=[output_ava, output_para, output_koniq, output_spaq])

optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

model.compile(optimizer=optimizer,
              loss=config.loss,
              metrics=[tf.keras.metrics.RootMeanSquaredError(), pearson_correlation]
              )

checkpoint_filepath = main_directory + 'multimodel_dataset/multimodel_checkpoint/'

# callbacks
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_Output_KonIQ_quality_pearson_correlation',
    mode='min'
    )

early_stopping_callback = EarlyStopping(
    monitor= config.metric2, 
    patience=config.early_patience,
    mode=config.early_mode,
    min_delta = config.early_min_delta
    )

reduce_lr_callback = ReduceLROnPlateau(
    monitor = config.metric2,
    factor = config.plateau_factor,
    patience = config.plateau_patience,
    mode = config.plateau_mode,
    min_lr = config.plateau_min_lr
)

lr_scheduler_callback = LearningRateScheduler(lr_schedule)

# Define the number of steps per epoch for each generator
steps_per_epoch1 = ava_train_generator.n//ava_train_generator.batch_size
steps_per_epoch2 = para_train_generator.n//para_train_generator.batch_size
steps_per_epoch3 = koniq_train_generator.n//koniq_train_generator.batch_size
steps_per_epoch4 = spaq_train_generator.n//spaq_train_generator.batch_size

val_steps1 = ava_val_generator.n//ava_val_generator.batch_size
val_steps2 = para_val_generator.n//para_val_generator.batch_size
val_steps3 = koniq_val_generator.n//koniq_val_generator.batch_size
val_steps4 = spaq_val_generator.n//spaq_val_generator.batch_size

test_steps1 = len(ava_test_generator)
test_steps2 = len(para_test_generator)
test_steps3 = len(koniq_test_generator)
test_steps4 = len(spaq_test_generator)

combined_train_gen = combined_generator(ava_train_generator, para_train_generator, koniq_train_generator, spaq_train_generator)

combined_val_gen = combined_generator(ava_val_generator, para_val_generator, koniq_val_generator, spaq_val_generator)

combined_test_gen = combined_generator(ava_test_generator, para_test_generator, koniq_test_generator, spaq_test_generator)

# train model
history = model.fit(x=combined_train_gen,
                    steps_per_epoch = max(steps_per_epoch1, steps_per_epoch2, steps_per_epoch3, steps_per_epoch4),
                    epochs = config.epoch,
                    validation_data = combined_val_gen,
                    validation_steps = max(val_steps1, val_steps2, val_steps3, val_steps4),
                    verbose=1,
                    callbacks = [
                      WandbMetricsLogger(log_freq=5),
                      WandbCallback(monitor='val_loss', mode='min', save_model=False, save_weights_only=False),
                      model_checkpoint_callback,
                      lr_scheduler_callback,
                      early_stopping_callback,
                      reduce_lr_callback
                    ])

wandb.finish()

# evaluate the model performance
evaluation = model.evaluate(
    combined_test_gen,
    steps=max(test_steps1, test_steps2, test_steps3, test_steps4)
)

saved_model_path = '/media/workstation/BackupDrive/model/multimodel_irnv2/'

model.save(saved_model_path + 'multimodel_irnv2_multimodel_full_train.h5')
model.save_weights(saved_model_path + 'multimodel_irnv2_weights_multimodel_full_train.h5')