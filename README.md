# Attention-Based-Chem-Predictor
Codebase for the paper "Application of Attention Mechanism-Based Neural Networks in Predicting Chemical Molecular Properties". Includes model training, evaluation, and data processing scripts.

[中文](README_zh.md)


## Project Structure

```
Attention-Based-Chem-Predictor/
│
├── edit-Attention-split.py       # Attention-based neural network model
├── edit-BP-split.py              # Backpropagation neural network model
├── edit-RBF-split.py             # Radial Basis Function neural network model
├── draw.m                        # MATLAB script for plotting loss curves
├── predictions.xlsx              
├── predictions_optimized.xlsx    
├── LICENSE                       
├── README.md                     
├── README_zh.md                     
│
├── history/                      # training loss history
│   ├── Attention_loss_history.txt
│   ├── BP_loss_history.txt
│   └── RBF_loss_history.txt
│
└── output/                       # output plots
    ├── Attention_loss_plot.png
    ├── BP_loss_plot.png
    └── RBF_loss_plot.png
```

## Description

This project implements and compares three different neural network models for predicting chemical molecular properties:

1. Attention-based Neural Network
2. Backpropagation Neural Network
3. Radial Basis Function Neural Network

Each model is implemented in a separate Python script and uses the same dataset (`data.csv`) for training and evaluation.

## Usage

Before running any scripts, ensure you have downloaded the `data.csv` file from the Releases section and placed it in the project's root directory.

### Python Scripts

To run any of the neural network models:

1. Ensure you have the required dependencies installed (pandas, numpy, tensorflow, scikit-learn).
2. Run the desired script:
```
   python edit-Attention-split.py
   python edit-BP-split.py
   python edit-RBF-split.py
```

Each script will:
- Load and preprocess the data from `data.csv`
- Split the data into training and testing sets
- Train the respective neural network model
- Evaluate the model and save predictions to `predictions.xlsx`
- Save the loss history to a text file in the `history/` directory

### MATLAB Script

To visualize the training and validation loss:

1. Open `draw.m` and set the `choose` variable to select the desired model (1 for Attention, 2 for RBF, 3 for BP).
2. Run the script.

The script will generate a loss plot and save it as a PNG file in the `output/` directory.

## Data

The `data.csv` file contains the dataset used for training and testing the models. It includes 100 feature columns and 3 target columns (y1, y2, y3).

Due to GitHub's file size limitations, the `data.csv` file is not included in the main repository. You can download it from the Releases section of this project. **Please download and place it in the root directory of the project before running the scripts.**

## Results

The training process generates loss history files in the `history/` directory. These files contain the training and validation loss for each epoch.

The `output/` directory contains PNG images of the loss curves for each model, generated using the MATLAB script.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.



