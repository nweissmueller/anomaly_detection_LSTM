import os
from src import libraries_and_config
from src import utils
from src import data_preparation
from src import model_definitions
from src import training_and_evaluation
from src import visualization

# Specify the data file path
data_filepath = os.path.join(os.getcwd(), 'data', 'raw', 'blood_lab_var1.csv')

# Load and prepare the data
df = data_preparation.load_and_prepare_data(data_filepath)

# ... (rest of your script for running the training, evaluation, and visualization)
