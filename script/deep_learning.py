# Create the network
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_model():
    new_model = keras.models.Sequential()
    new_model.add(keras.layers.Dense(64, activation='relu', input_shape=(9,)))
    new_model.add(keras.layers.Dropout(0.5))
    new_model.add(keras.layers.Dense(48, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    new_model.add(keras.layers.Dropout(0.5))
    new_model.add(keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    new_model.add(keras.layers.Dropout(0.5))
    new_model.add(keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    new_model.add(keras.layers.Dropout(0.5))
    new_model.add(keras.layers.Dense(1, activation='sigmoid'))
    new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',
                      metrics=['accuracy'])
    return new_model


def cross_validation(x_train, y_train):
    max_validation_score = 0
    k = 6
    best_model = None
    best_history = None
    num_samples = len(x_train) // k
    validation_scores = []
    for fold in range(k):
        validation_data = x_train[num_samples * fold:num_samples * (fold + 1)]
        validation_label = y_train[num_samples * fold:num_samples * (fold + 1)]
        training_data = np.concatenate((x_train[:num_samples * fold], x_train[num_samples * (fold + 1):]))
        training_label = np.concatenate((y_train[:num_samples * fold], y_train[num_samples * (fold + 1):]))
        model = get_model()
        history = model.fit(training_data, training_label, epochs=100,
                            validation_data=(validation_data, validation_label))
        validation_score = model.evaluate(validation_data, validation_label)
        if validation_score[1] > max_validation_score:
            max_validation_score = validation_score[1]
            best_history = history
            best_model = model
        validation_scores.append(validation_score)

    validation_score = np.average(validation_scores)
    print(validation_score)
    return best_history, best_model


def plot_loss(history):
    plt.subplots(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.draw()
    plt.show()


def plot_accuracy(history):
    plt.subplots(figsize=(12, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.draw()
    plt.show()


def plot_conf_matr(model, x_test, y_test):
    predictions = model.predict(x_test)
    classes = predictions > 0.5
    cm = confusion_matrix(y_test, classes)

    # Plot
    plt.figure(figsize=(10, 7))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax,
                cmap="PuBu")  # annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['No Potable', 'Potable'])
    ax.yaxis.set_ticklabels(['No Potable', 'Potable'])
    print(classification_report(y_test, classes))
    plt.draw()
    plt.show()
