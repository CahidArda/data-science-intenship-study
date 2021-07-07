# ATM Demand Forecasting

In this project, I worked on forecasting cash demand for ATMs. I began with exploring the data to gain insight. With this insight, I planned a feature set and implemented methods to create this dataset. Finally, I used the feature set I created to train models for forecasting aggregate and individual atm demand.

## Notebooks

### `atm_eda.ipynb`

* Used pandas to import the dataset.
* Used matplotlib and plotly to visualize the data
* Cleaned the data by removing outliers
* Investigated spikes in data to gain insight
* Observed the correlation over time

### `atm_fe.ipynb`

* Planned a feature set
* Implemented functions to create the planned feature set.
* Transferred all the feature generation methods to `feature_generation.py`

### `atm_forecasting.ipynb`

* Implemented MAPE error
* Calculated base scores to evaluate accuracy of trained models
* Made predictions with Random Forest
* Used Optuna to find hyper-parameters for the LGBM model 
* Trained LGBM models using the hyper-parameters and KFold
* Plotted the error over time, actual values and predictions using a helper function
* Transferred all the forecasting and visualization methods to `forecating.py`

### `atm_individual_forecasting.ipynb`

* Applied methods in `atm_forecasting.ipynb` to the data of a single ATM unit instead of aggregating.