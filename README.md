# Description

[MAHOMES](https://github.com/SluskyLab/MAHOMES) is a project done by the SLUSKY lab in the University of Kansas to distinguish between active and inactive metal-binding sites of proteins. The methodology is a variety of machine learning algorithms/tools that can be used individually, but haven't been designed in tandem. Overall, the models performed relatively well to other previous approaches, with 92.2% precision and 901% recall. One of their algorithms, MLP (Multilayer Perceptron), uses scikit-learn’s Multilayer Perceptron library. However, it happened to do relatively worse in MCC (Matthews Correlation Coefficient) and precision measures compared to other methods, but by 1 to 2 percent (no major gap in performance). Scikit-learn’s MLP is often used as a very surface-level MLP that doesn’t have many customization options, but has the advantage of near guaranteeing convergence, in this case, outputs of 0 and 1s for this binary classification. 
This project involves a lot of trial and error with using Pytorch and Tensorflow MLP model builders. The goal is to design a custom MLP that can be directly edited to adjust hidden layer features and activation functions, allowing more customization than scikit learn. In addition, while the model may not perform as well as scikit learn, it will achieve a decent accuracy and true negative rate in relation to MAHOMES scikit learn MLP, and there is plenty opportunitiy to further customize the model to improve the accuracy.

## Dataset Selection

The dataset is found in the data folder, in the file sites_calculated_features.txt. The other two csv files are scaled training and test versions of that overall data.
The data was borrowed from the MAHOMES github repository, and it contains various protein sites, their physical featurs (tabular values), and their catalytic/non-catalytic status as true/false. The catalytic status is the target of our machine learning.

The direct process by which the data was obtained can be found in the [MAHOMES I paper](https://www.nature.com/articles/s41467-021-24070-3). It was sourced from RCSB, filtered for quality and with metal atoms of similar crystal structure grouped together as one site. The most high quality data was sourced, redundant data filtered, and only unique residues of each site were represented in the data. Feature distribution is good (no lack of pocket and lining features, good representatin in the data) Overall, the dataset is a bit small (around 4000 total unique sites), and around 24% of the data is catalytic. 
The evaluation of the new custom MLP in comparison to MAHOMES MLP was done by tensorflow's confusion matrix and the Matthews Correlation Coefficient, among a few other evaluated parameters.



# Results:


# Credits

All credit of utils.py to MAHOMES project (https://github.com/SluskyLab/MAHOMES) 
