# -*- coding: utf-8 -*-
"""Water-Potability """
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from variables import *
from keras.utils.vis_utils import plot_model
from analyze_dataset import *
from deep_learning import *

if __name__ == "__main__":

    # Read Files: In this section we import the csv files.
    df = pd.read_csv('water_potability.csv')

    # Split the dataset into label and data
    labels = df['Potability']
    data = df.drop(['Potability'], axis=1)

    # Analyze Dataset: In this section we will analyze the dataset shape, balance and if it has null values in its rows

    # Shape: Check the shape of the dataset
    print('Data shape:', data.shape)
    print('Labels shape:', labels.shape)

    # Null values: Check if there are null values
    num_null = df.isnull().sum(axis=0).sum()
    fix_null = False
    if num_null > 0:
        fix_null = True

    # Balance: Check if the dataset is balanced
    balanced = True
    zero_percentage = balance(df['Potability'])
    if zero_percentage != 50.0:
        balanced = False
    
    # Outliers, Skewness and Correlation
    # Outliers
    plot_outliers(df, False)
    # Skewness
    plot_skewness(df)
    # Correlation
    plot_correlation(df)

    # Fix Null Values
    if fix_null:
        if solution == "mean":
            df.ph = df.ph.fillna(df.ph.mean())
            df.Sulfate = df.Sulfate.fillna(df.Sulfate.mean())
            df.Trihalomethanes = df.Trihalomethanes.fillna(df.Trihalomethanes.mean())
        elif solution == "median":
            df.ph = df.ph.fillna(df.ph.median())
            df.Sulfate = df.Sulfate.fillna(df.Sulfate.median())
            df.Trihalomethanes = df.Trihalomethanes.fillna(df.Trihalomethanes.median())
        else:
            df = df.dropna()

    # Fix outliers
    if substitute:
        df = capping_flooring(df)
        plot_outliers(df, True)

    # Normalize values
    df = normalize_dataset(df)

    # Update labels and data
    labels = df['Potability']
    data = df.drop(['Potability'], axis=1)

    # Split data into training, validation and test set
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.9)
    if solution == "drop":
        x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.8)

    if not balanced:
        sm = SMOTE()
        x_train, y_train = sm.fit_resample(x_train, y_train)
        print('Train data shape after balance:', x_train.shape)
        print('Train labels shape after balance:', y_train.shape)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8)

    if solution == "drop":
        # Print all the sizes
        print('Train data shape:', x_train.shape)
        print('Train labels shape:', y_train.shape)
        print('Test data shape:', x_test.shape)
        print('Test labels shape:', y_test.shape)
        input("PRESS ENTER TO CONTINUE.")
    else:
        # Print all the sizes
        print('Train data shape:', x_train.shape)
        print('Train labels shape:', y_train.shape)
        print('Validation data shape:', x_valid.shape)
        print('Validation labels shape:', y_valid.shape)
        print('Test data shape:', x_test.shape)
        print('Test labels shape:', y_test.shape)
        input("PRESS ENTER TO CONTINUE.")

    if not load_model:

        model = get_model()

        # Train the network
        history = None
        if train_model and solution != "drop":
            history = model.fit(x_train, y_train, epochs=200, validation_data=(x_valid, y_valid))

        # Cross Validation
        if solution == "drop":
            history, model = cross_validation(x_train, y_train)

        # Loss graph of the model
        if model_loss:
            plot_loss(history)

        # Accuracy graph of the model
        if model_accuracy:
            plot_accuracy(history)
    else:
        # Load model
        model = keras.models.load_model(file_name)

    # Evaluate the model: Check how well the dataset perform on the test set
    if evaluate_model:
        model.evaluate(x_test, y_test)

    # Confusion Matrix: Compute the label prediction using the test set and plot the confusion matrix.
    if conf_matr:
        plot_conf_matr(model, x_test, y_test, "Confusion Matrix")

    # Save the model
    if save_model:
        model.save(file_name)

    # Plot model
    if plot_model:
        dot_img_file = "network.png"
        keras.utils.model_to_dot(model, show_shapes=True)
