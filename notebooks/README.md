# Notebooks

## ALU

I attempted to code and train a new layer. Explanation is available in `alu` notebook

## ATM_DEMAND

This file has notebooks focused on data science and tree-based machine learning methods. I did the following in these notebooks:
* `atm_eda`: Basic exploratory data exploration
* `atm_fe`: Feature engineering to create a dataset from the insight I got from atm_eda and model testings.
* `atm_forecasting`: Forecasting with the aggregate of the dataset
* `atm_individual_forecasting`: Forecasting using a single ATM data instead of aggregating over many

See the README file in ATM_DEMAND folder for more information about the notebooks.

There are also some `.py`. These files are created with the code in the notebooks for future use.

## RNN

This file has notebooks focused on RNN's and other deep learning based machine learning methods. I did the following in these notebooks:
* `tf_custom_rnn`: Implemented an RNN layer with tensorflow
* `tf_custom_rnn_forecasting`: Used the custom RNN layer implemented in `tf_custom_rnn` notebook to make predictions
* `keras_rnn_dense`: Used keras implementation of RNN to make predictions
* `keras_pp_dense_forecasting`: Instead of using an RNN, used preprocessing and dense layers to make predictions. This model is similar to a TabMlp.
* `keras_pp_transformer_dense_forecasting`: Added transformer layers to the model to implement a TabTransformer layer. 