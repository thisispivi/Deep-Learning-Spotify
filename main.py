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
    df = pd.read_csv(dataset_file)

    # Split the dataset into label and data
    labels = df['Potability']
    data = df.drop(['Potability'], axis=1)

    # Analyze Dataset: In this section we will analyze the dataset shape, balance and if it has null values in its rows

    # Shape: Check the shape of the dataset
    print('\nCheck Dataset Shape')
    print('Data shape:', data.shape)
    print('Labels shape:', labels.shape)

    # Null values: Check if there are null values
    print('\nCheck Dataset Null Values')
    num_null = df.isnull().sum(axis=0).sum()
    fix_null = False
    if num_null > 0:
        fix_null = True
        print("There are null values in the dataset\n")
    else:
        print("There aren't null values in the dataset\n")

    # Balance: Check if the dataset is balanced
    print("Check if the Dataset is balanced")
    balanced = True
    if not save_figure:
        zero_percentage = balance(df['Potability'], False)
    else:
        zero_percentage = balance(df['Potability'], False, img_folder /
                                  "class_balance_bar.png", img_folder / "class_balance_pie.png")
    if zero_percentage != 50.0:
        print("Data is not Balanced")
        balanced = False
    else:
        print("Data is Balanced")

    ### Outliers, Skewness and Correlation

    # Outliers
    if not save_figure:
        plot_outliers(df, False)
    else:
        plot_outliers(df, False, img_folder / "boxplot.png")

    # Skewness
    if not save_figure:
        plot_skewness(df)
    else:
        plot_skewness(df, img_folder / "distplot.png")

    # Correlation
    if not save_figure:
        plot_correlation(df)
    else:
        plot_correlation(df, img_folder / "correlation.png")

    # Fix Null Values
    if fix_null:
        print("\nFix Null values")
        if solution == "mean":
            df.ph = df.ph.fillna(df.ph.mean())
            df.Sulfate = df.Sulfate.fillna(df.Sulfate.mean())
            df.Trihalomethanes = df.Trihalomethanes.fillna(
                df.Trihalomethanes.mean())
        elif solution == "median":
            df.ph = df.ph.fillna(df.ph.median())
            df.Sulfate = df.Sulfate.fillna(df.Sulfate.median())
            df.Trihalomethanes = df.Trihalomethanes.fillna(
                df.Trihalomethanes.median())
        else:
            df = df.dropna()

    # Fix outliers
    if substitute:
        print("\nFix Outliers")
        df = capping_flooring(df)
        if not save_figure:
            plot_outliers(df, False)
        else:
            plot_outliers(df, False, img_folder / "boxplot_after.png")

    # Normalize values
    print("\nNormalize Dataset\n")
    df = normalize_dataset(df)

    # Update labels and data
    labels = df['Potability']
    data = df.drop(['Potability'], axis=1)

    # Split data into training, validation and test set
    print("Split training and test set\n")
    if solution == "drop":
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, train_size=0.8)
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, train_size=0.9)

    if not balanced:
        print("Balance Dataset using SMOTE")
        sm = SMOTE()
        x_train, y_train = sm.fit_resample(x_train, y_train)
        print('Train data shape after balance:', x_train.shape)
        print('Train labels shape after balance:', y_train.shape)

        if not save_figure:
            balance(pd.DataFrame(y_train), True)
        else:
            balance(pd.DataFrame(y_train), True, img_folder /
                    "class_balance_bar_post.png", img_folder / "class_balance_pie_post.png")

    if solution != "drop":
        print("\nSplit Training set into Validation and Training set")
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, train_size=0.8)

    if solution == "drop":
        # Print all the sizes
        print('Train data shape:', x_train.shape)
        print('Train labels shape:', y_train.shape)
        print('Test data shape:', x_test.shape)
        print('Test labels shape:', y_test.shape)
    else:
        # Print all the sizes
        print('Train data shape:', x_train.shape)
        print('Train labels shape:', y_train.shape)
        print('Validation data shape:', x_valid.shape)
        print('Validation labels shape:', y_valid.shape)
        print('Test data shape:', x_test.shape)
        print('Test labels shape:', y_test.shape)

    if not load_model:
        print("\nCreate new model")
        model = get_model()

        # Train the network
        history = None
        if train_model and solution != "drop":
            print("\nTrain the network")
            history = model.fit(x_train, y_train, epochs=200,
                                validation_data=(x_valid, y_valid))

        # Cross Validation
        if solution == "drop":
            print("\nCross validation")
            history, model = cross_validation(x_train, y_train)

        # Loss graph of the model
        if model_loss:
            if not save_figure:
                plot_loss(history)
            else:
                plot_loss(history, img_folder / "model_loss.png")

        # Accuracy graph of the model
        if model_accuracy:
            if not save_figure:
                plot_accuracy(history)
            else:
                plot_accuracy(history, img_folder / "model_accuracy.png")
    else:
        # Load model
        print("\nLoad model")
        model = keras.models.load_model(file_name)

    # Evaluate the model: Check how well the dataset perform on the test set
    if evaluate_model:
        print("\nModel Performance")
        model.evaluate(x_test, y_test)

    # Confusion Matrix: Compute the label prediction using the test set and plot the confusion matrix.
    if conf_matr:
        if not save_figure:
            plot_conf_matr(model, x_test, y_test, 'Confusion Matrix')
        else:
            plot_conf_matr(model, x_test, y_test,
                           'Confusion Matrix', img_folder / "conf_matr.png")

    # Save the model
    if save_model:
        print("\nSave Model")
        model.save(file_name)
