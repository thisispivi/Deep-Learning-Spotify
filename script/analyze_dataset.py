import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler


def balance(new_df, smote):
    result = new_df.value_counts()
    zero_percentage = round((result[0] * 100) / (result[0] + result[1]), 2)
    print("No. of 0: " + str(result[0]) + "\nNo. of 1: " + str(result[1]) +
          "\nPercentage of 0: " + str(zero_percentage) + " %\nPercentage of 1: " +
          str(round((100 - zero_percentage), 2)) + " %")

    plt.bar(x=["No Potability", "Potability"], height=[result[0], result[1]], color=["royalblue", "indianred"])
    plt.ylabel("Count")
    if smote:
        plt.title("Number of No Potability rows vs number of Potability rows / Balanced Dataset")
    else:
        plt.title("Number of No Potability rows vs number of Potability rows / Original Dataset")
    plt.draw()
    plt.show()

    plt.pie([result[0], result[1]], labels=["No Potability", "Potability"], explode=(0.1, 0), autopct='%1.2f%%',
            colors=["thistle", "paleturquoise"], radius=1.2)
    if smote:
        plt.title("Number of No Bankrupt rows vs number of Bankrupt rows / Balanced Dataset")
    else:
        plt.title("Number of No Bankrupt rows vs number of Bankrupt rows / Original Dataset")

    plt.draw()
    plt.show()
    return zero_percentage


def plot_outliers(df, without_outlier):
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    if without_outlier:
        fig.suptitle('Without outliers after capping and flooring')
    else:
        fig.suptitle('Checking Outliers')
    sns.boxplot(ax=ax[0, 0], x=df["ph"], data=df)
    sns.boxplot(ax=ax[0, 1], x=df["Hardness"], data=df)
    sns.boxplot(ax=ax[0, 2], x=df["Solids"], data=df)
    sns.boxplot(ax=ax[1, 0], x=df["Chloramines"], data=df)
    sns.boxplot(ax=ax[1, 1], x=df["Sulfate"], data=df)
    sns.boxplot(ax=ax[1, 2], x=df["Conductivity"], data=df)
    sns.boxplot(ax=ax[2, 0], x=df["Organic_carbon"], data=df)
    sns.boxplot(ax=ax[2, 1], x=df["Trihalomethanes"], data=df)
    sns.boxplot(ax=ax[2, 2], x=df["Turbidity"], data=df)
    plt.draw()
    plt.show()


def plot_skewness(df):
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Checking Skewness')
    sns.distplot(ax=ax[0, 0], x=df["ph"][df["Potability"] == 1], color='cyan', axlabel="ph")
    sns.distplot(ax=ax[0, 0], x=df["ph"][df["Potability"] == 0], color='violet', axlabel="ph")
    ax[0, 0].legend(['Potable', 'Not Potable'])
    sns.distplot(ax=ax[0, 1], x=df["Hardness"][df["Potability"] == 1], color='cyan', axlabel="Hardness")
    sns.distplot(ax=ax[0, 1], x=df["Hardness"][df["Potability"] == 0], color='violet', axlabel="Hardness")
    ax[0, 1].legend(['Potable', 'Not Potable'])
    sns.distplot(ax=ax[0, 2], x=df["Solids"][df["Potability"] == 1], color='cyan', axlabel="Solids")
    sns.distplot(ax=ax[0, 2], x=df["Solids"][df["Potability"] == 0], color='violet', axlabel="Solids")
    ax[0, 2].legend(['Potable', 'Not Potable'])
    sns.distplot(ax=ax[1, 0], x=df["Chloramines"][df["Potability"] == 1], color='cyan', axlabel="Chloramines")
    sns.distplot(ax=ax[1, 0], x=df["Chloramines"][df["Potability"] == 0], color='violet', axlabel="Chloramines")
    ax[1, 0].legend(['Potable', 'Not Potable'])
    sns.distplot(ax=ax[1, 1], x=df["Sulfate"][df["Potability"] == 1], color='cyan', axlabel="Sulfate")
    sns.distplot(ax=ax[1, 1], x=df["Sulfate"][df["Potability"] == 0], color='violet', axlabel="Sulfate")
    ax[1, 1].legend(['Potable', 'Not Potable'])
    sns.distplot(ax=ax[1, 2], x=df["Conductivity"][df["Potability"] == 1], color='cyan', axlabel="Conductivity")
    sns.distplot(ax=ax[1, 2], x=df["Conductivity"][df["Potability"] == 0], color='violet', axlabel="Conductivity")
    ax[1, 2].legend(['Potable', 'Not Potable'])
    sns.distplot(ax=ax[2, 0], x=df["Organic_carbon"][df["Potability"] == 1], color='cyan', axlabel="Orgainc Carbon")
    sns.distplot(ax=ax[2, 0], x=df["Organic_carbon"][df["Potability"] == 0], color='violet', axlabel="Orgainc Carbon")
    ax[2, 0].legend(['Potable', 'Not Potable'])
    sns.distplot(ax=ax[2, 1], x=df["Trihalomethanes"][df["Potability"] == 1], color='cyan', axlabel="Trihalomethanes")
    sns.distplot(ax=ax[2, 1], x=df["Trihalomethanes"][df["Potability"] == 0], color='violet', axlabel="Trihalomethanes")
    ax[2, 1].legend(['Potable', 'Not Potable'])
    sns.distplot(ax=ax[2, 2], x=df["Turbidity"][df["Potability"] == 1], color='cyan', axlabel="Turbidity")
    sns.distplot(ax=ax[2, 2], x=df["Turbidity"][df["Potability"] == 0], color='violet', axlabel="Turbidity")
    ax[2, 2].legend(['Potable', 'Not Potable'])

    plt.draw()
    plt.show()


def plot_correlation(df):
    plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, linewidth=.01, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Correlation")
    plt.draw()
    plt.show()


def capping_flooring(df):
    for col in df:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        whisker_width = 1.5
        lower_whisker = q1 - (whisker_width * iqr)
        upper_whisker = q3 + (whisker_width * iqr)
        df[col] = np.where(df[col] > upper_whisker, upper_whisker,
                           np.where(df[col] < lower_whisker, lower_whisker, df[col]))
    return df


def normalize_dataset(df):
    # Take the columns with values over 1
    cols_for_scale = df.max()[df.max() > 1]
    # Take the columns with values less than 0
    var = df.min()[df.min() < 0]  # It is none there aren't negative values
    # Normalize values
    scale = StandardScaler()
    scaled = scale.fit_transform(df[cols_for_scale.keys()])
    # Substitute the old values with the normalized ones
    i = 0
    for column in cols_for_scale.keys():
        df[column] = scaled[:, i]
        i += 1
    return df