# Recurrent Neural Networks

## Setting up Tensorflow Environment in Conda

Create a new environment named 'tensor':
```
conda create -n tensor python=3.7
```
Activate the new environment:
```
conda activate tensor
```
To make VS Code connect with the conda environment:
```
conda install ipykernel
```
After checking that VS Code can connect with the environment, install tensorflow:
```
conda install tensorflow
conda install matplotlib
conda install pandas
conda install -c conda-forge scikit-learn
conda install -c plotly plotly
```