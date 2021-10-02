from pathlib import Path

"""
Null values management -> Uncomment to select one
"""
# solution = "mean"
solution = "median"
# solution = "drop"


"""
Configure the variables
        
train_model -> True: the network will be trained / False: network won't be trained
model_loss -> True: plot the model loss / False: don't plot the model loss
model_accuracy -> True: plot the model accuracy / False: don't plot the model accuracy
evaluate_model -> True: evaluate the model / False: don't evaluate the model
conf_matr -> True: plot the confusion matrix / False: don't plot the confusion matrix
plot_model -> True: plot the structure of the network / False: don't plot the structure of the network
save_model -> True: save the model / False: don't save the model
load_model -> True: load a model in the network folder / False: don't load the model
save_figure -> True: save the image of the plots / False: show the plots
substitute  -> True: substitute outliers / False: don't fix outliers    
"""

train_model = True
model_loss = True
model_accuracy = True
evaluate_model = True
conf_matr = True
plot_model = False
save_model = True
load_model = False
save_figure = True
substitute = True

data_folder = Path(__file__).parent.resolve() / Path("data")
dataset_file = data_folder / 'water_potability.csv'

save_folder = Path(__file__).parent.resolve() / Path("network")
file_name = save_folder / 'new'

img_folder = Path(__file__).parent.resolve() / Path("figures")
