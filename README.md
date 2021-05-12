# deepALForCalibration

In this experiment, we try to see if and how diversity sampling strategy reduces the calibration error of machine learning models. We further compare the performance of diversity sampling to uncertainty sampling and random sampling. We use the scikit-learn's MLP model with the optimal parameters we found via hyperparameter search.

"SearchHyperParams.ipynb": You can run this notebook to find the optimal parameters for your dataset
"data_resampling_for_imbalanced_binary_classification": You can run this notebook to resample your dataset
"ALforCalibration_MLP.ipynb": The main notebook to test different active learning strategies with MLP model. We run this notebook on Colab, so if you run in your local, you should comment out the cell corresponding to the Colab mounting. On the 4th code cell, you need to define the path for your dataset, and all the parameters you want to use. Comments on to notebook will guide you.
