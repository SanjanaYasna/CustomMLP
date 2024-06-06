from scikeras.wrappers import KerasClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#to resolve missing distutils if your python is most recent:
import setuptools.dist 
#other crap
import tensorflow as tf
from kerastuner.tuners import RandomSearch
import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class KerasMLP:
    #model initialization
    def init_model(init_features):
        inputs = keras.Input(shape=(init_features,), name = "inputs")
        #first relu
        actv1 = keras.layers.Dense(init_features, activation="relu")(inputs)
        layer1 = keras.layers.Dense(init_features, activation="relu")(actv1)
        #second relu
        actv2 = keras.layers.Dense(init_features, activation="relu")(layer1)
        layer2 = keras.layers.Dense(90, activation="relu")(actv2)
        #sigmoid activation
        actv3 = keras.layers.Dense(90, activation="relu")(layer2)
        output = keras.layers.Dense(1, activation="sigmoid", name = "outputs")(actv3) #shoudl be one binary output, hopefully
        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
        #result metrics for keras (also in utilties) Accuracy, MCC, Recall, TrueNegRate, Precision
    def check_result_metrics_for_keras(alg, feat_set, prediction_df):
        mcc = matthews_corrcoef(prediction_df['actual'], prediction_df['bool_pred'])
        TN, FP, FN, TP = confusion_matrix(prediction_df['actual'], prediction_df['bool_pred']).ravel()

        TPR=(TP/(TP+FN))*100
        TNR=(TN/(TN+FP))*100
        acc=((TP+TN)/(TP+TN+FP+FN))*100
        Prec=(TP/(TP+FP))*100
        return(pd.DataFrame([[alg, feat_set, acc, mcc, TPR, TNR, Prec]],
            columns=['Algorithm', 'Feature Set', 'Accuracy', 'MCC', 'Recall', 'TrueNegRate', 'Precision']))
