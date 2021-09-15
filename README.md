# Deep-Learning-Water-Potability
The objective of the project is to perform the classification of water quality using Deep Learning. The dataset is taken from [kaggle](https://www.kaggle.com/adityakadiwal/water-potability) and it contains the water quality metrics of 3276 water bodies.

This readme will explain the dataset structure, how the project works and the best network achieved so far.


# Index

- [Project structure](#project-structure)
- [Dataset structure](#dataset-structure)
  * [1. pH value:](#1-ph-value-)
  * [2. Hardness:](#2-hardness-)
  * [3. Solids (Total dissolved solids - TDS):](#3-solids--total-dissolved-solids---tds--)
  * [4. Chloramines:](#4-chloramines-)
  * [5. Sulfate:](#5-sulfate-)
  * [6. Conductivity:](#6-conductivity-)
  * [7. Organic_carbon:](#7-organic-carbon-)
  * [8. Trihalomethanes:](#8-trihalomethanes-)
  * [9. Turbidity:](#9-turbidity-)
  * [10. Potability:](#10-potability-)
- [How the project works](#how-the-project-works)
  * [Dataset download and import](#dataset-download-and-import)
  * [Dataset analysis](#dataset-analysis)
    + [Shape](#shape)
    + [Null Values](#null-values)
    + [Balance](#balance)
    + [Check Values](#check-values)
  * [Fix Null Values](#fix-null-values)
  * [Data Normalization](#data-normalization)
  * [Balance Dataset](#balance-dataset)
  * [Split data into training, validation and test set](#split-data-into-training--validation-and-test-set)
  * [Network creation](#network-creation)
    + [Options](#options)
    + [Network Creation and Training](#network-creation-and-training)
    + [Network Evaluate](#network-evaluate)
    + [Save Model](#save-model)
  * [Load Model](#load-model)
- [Best model analysis](#best-model-analysis)
  * [Network structure](#network-structure)
  * [Model Loss](#model-loss)
  * [Model Accuracy](#model-accuracy)
  * [Test set performance](#test-set-performance)
    + [Confusion Matrix](#confusion-matrix)

# Project structure
```
.
|
| Folders
├── data   # Folder with the dataset
│   └── data.zip
├── network   # Folder with the network
│   └── network.zip
├── img   # Images of the graphs for the analysis report
|
| Notebook
└── Water_Potability.ipynb
```

# Dataset structure

The ```water_potability.csv``` file contains water quality metrics for 3276 different water bodies.

## 1. pH value:
PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.

## 2. Hardness:
Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.

## 3. Solids (Total dissolved solids - TDS):
Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.

## 4. Chloramines:
Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.

## 5. Sulfate:
Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.

## 6. Conductivity:
Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.

## 7. Organic_carbon:
Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.

## 8. Trihalomethanes:
THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.

## 9. Turbidity:
The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.

## 10. Potability:
Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.

# How the project works

This section will show how the project works.

## Dataset download and import
In the first part of the code there will be the download of the dataset from github or from kaggle. In the second case it's important to insert in the corret folder the api token. In this [link](https://www.kaggle.com/docs/api) there's a guide on how to create a token.

Next using pandas the dataset will be inserted in a pandas dataframe.

## Dataset analysis
In the next part there will be an analysis of the dataset.

### Shape
The first thing done is to check the shape of the dataframe: **(3276,10)**

### Null Values
The second thing is to check if there are null values. 

```
ph                 491
Hardness             0
Solids               0
Chloramines          0
Sulfate            781
Conductivity         0
Organic_carbon       0
Trihalomethanes    162
Turbidity            0
Potability           0
```

Unortunately there are three columns with null values.

### Balance
Next it's important to check the balance of the dataset, because if there is a class that has more rows than the other the classification will have a good accuracy but it won't perform well on the minor class.

| Class | Number | Percentage |
|:-----:|:------:|:----------:|
|   0   |  1998  |   60.99 %  |
|   1   |  1278  |   39.01 %   |

![class_balance_bar](img/class_balance_bar.png)

![class_balance_pie](img/class_balance_pie.png)

So the dataset is unbalanced, the potability class is just 39.01%. This means that there will be a step in which using SMOTE the dataset will be balanced.

### Check Values
It is important also to check if all the data are normalized. So if the code finds some values that are bigger than 1 and lower than 0, a normalization step will be performed. Unfortunately the dataset isn't all normalized.

## Fix Null Values

In this section we will fill the null values of the columns: ph, sulfate and trihalomethanes. 

```
df.ph = df.ph.fillna(df.ph.mean())
df.Sulfate = df.Sulfate.fillna(df.Sulfate.mean())
df.Trihalomethanes = df.Trihalomethanes.fillna(df.Trihalomethanes.mean())
```

We will fill the null values with the columns mean.

## Data Normalization
In this section there will be the nomalization of the values. This process will use the ```StandardScaler()```. This scaler uses the mean and the standard deviation to set all values to between 0 and 1.

## Balance Dataset
The dataset has been balanced using [**SMOTE**](https://towardsdatascience.com/applying-smote-for-class-imbalance-with-just-a-few-lines-of-code-python-cdf603e58688) (Synthetic Minority Oversampling Technique).

The dataset will be filled with new data and it will be balanced.

The new shape is **(3996,10)**

| Class | Number | Percentage |
|:-----:|:------:|:----------:|
|   0   |  1988  |    50.0 %  |
|   1   |  1988  |    50.0 %  |

![class_balance_bar_post](img/class_balance_bar_post.png)

![class_balance_pie_post](img/class_balance_pie_post.png)

Now the dataset can be used with the networks.

## Split data into training, validation and test set

Split the data in:
* ```x_train```: The training set data
* ```y_train```: The training set label
* ```x_valid```: The validation set data
* ```y_valid```: The validation set label
* ```x_test```: The validation set data
* ```y_test```: The validation set label

The dimension will be something like

| Set | Percentage | Rows |
|:---:|:----------:|:----:|
|Training| 70 % | 2876 |
| Validation | 20 % | 720 |
| Test | 10 % | 400 |

## Network creation
In the code there is a section called **Create New Model** and it is helpful to create, train and evaluate a model.

### Options
In the first part of the section there are some boolean variables that tune what the code will do:

* ```train_model``` -> True: the network will be trained / False: network wont' be trained
* ```model_loss``` -> True: plot the model loss / False: don't plot the model loss
* ```model_accuracy``` -> True: plot the model accuracy / False: don't plot the model accuracy
* ```evaluate_model``` -> True: evaluate the model / False: don't evaluate the model
* ```conf_matr``` -> True: plot the confusion matrix / False: don't plot the confusion matrix
* ```plot_model``` -> True: plot the structure of the network / False: don't plot the structure of the network
* ```save_model``` -> True: save the model / False: don't save the model

### Network Creation and Training
After the variables tuning there is the network creation and training. 

### Network Evaluate
In this section there is the network evaluation. The code will plot useful data to understand how well the model is made and how it performs on the test set.

The plots will be:
* The model loss graph
* The model accuracy graph
* The performance of the test set
* The confusion matrix

### Save Model
At the end there is the possibility to save the model into a zip file. The only parameters to configure are the name of the folder *file_name* and the name of the folder in the zip command.

## Load Model

There is also a section to load a model. To do this it's important to follow these steps:

1. Load into colab the ```model.zip``` file
2. Uncomment the load data section
3. Insert the name of the folder into the *file_name* variable
4. Run the load data section

# Best model analysis
In this section we will analyze the best model that we achieved.

## Network structure

The network used has this structure:

![Network](img/network.png)

For each Dense layer except the last one there is **relu** as activation function. In the last Dense layer there is the **sigmoid** activation function.

In all Dense layer in the middle of the network there also the **l2 kernel regularizer** setted with (0.001).

The optimizer is **Adam** with the learning rate set at 0.001.

The **loss function** is the binary crossentropy.

We trained the network for 200 epochs.

```python
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(9,)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(48,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(32,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',metrics=['accuracy'])
```

## Model Loss
![Model Loss](img/model_loss.png)

As we can see there are some spikes in the Validation but overall it follows the Training loss. So there is no underfitting and no overfitting.

## Model Accuracy

![Model Accuracy](img/model_accuracy.png)

As we can see there are some spikes in the accuracy and as we can see there's a little overfitting. 

## Test set performance

In this section we will see how well the network perform on the training set.

| Accuracy | Loss |
|:--------:|:----:|
| 68.75 % | 0.5980 |

0.6283611059188843, 0.6725000143051147

| Class | Precision | Recall | f1-score | support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| 0 | 0.68 | 0.71 | 0.70 | 202 |
| 1 | 0.69 | 0.67 | 0.68 | 198 |

### Confusion Matrix

![Conf_Matr](img/conf_matr.png)

As we can se only many values were misclassified.