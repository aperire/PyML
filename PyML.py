### itertools ###
import itertools
### numpy ###
import numpy as np
### pandas ###
import pandas as pd
### seaborn ###
import seaborn as sns
### matplotlib ###
import matplotlib.pyplot as plt
### sklearn ###
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
### pytorch ###
import torch
import torch.nn.functional as F
from torch import optim, nn

class DataProcessing:
    def __init__(self, filepath):
        self.filepath = filepath

    def csv_to_DataFrame(self, drop):  # drop is a bool where you decide to drop NaN values or fill with 0 :: True = dropna, False = fillna. 
        data = pd.read_csv(self.filepath)
        df = pd.DataFrame(data)
        if drop == True:
            df.dropna()
        elif drop == False:
            df.fillna(0)
        else:
            raise TypeError("csv_to_DataFrame accepts bool as an argument!")
        return df  # returns dataframe

    def df_to_dict(self, df):  # accepts DataFrame, converts it into a dict
        raw_dict = df.to_dict()
        fin_dict = {}
        for title, group in raw_dict.items():
            fin_dict[title] = list(group.values())
        return fin_dict
    
    def to_dummies(self, df, lst_dummies):  # accepts DataFrame and lst_dummies(contains columns to convert)
        for i in lst_dummies:
            df = pd.get_dummies(df, columns=[i])
        return df

    def to_floats(self, df, lst_floats):  # accepts DataFrame and lst_floats(contains columns to convert) converts values to type float
        for i in lst_floats:
            df[i] = df[i].astype(float)
        return df

    def feature_label_split(self, df, feature_lst, label_lst):  # accepts DataFrame and list of features and label
        feature = df[feature_lst]
        label = df[label_lst]      
        return feature, label  # returns dataframe of feature and label

    def train_test_split(self, feature, label, train_size=0.8, random_state=None):  # accepts DataFrame of feature, label and float of training size. Can use randomstate
        feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size = train_size, random_state=random_state)
        return feature_train, feature_test, label_train, label_test  # returns splitted val
    

class MachineLearning:
    def __init__(self, feature_train, feature_test, label_train, label_test):
        self.feature_train = feature_train
        self.feature_test = feature_test
        self.label_train = label_train
        self.label_test = label_test

    def linear_regression(self, df, graph=True, title="Linear Regression", predict_features={}, display=False):  # DF should be given. accepts graph (bool) if True, plot graph, showing progress of training. Set title. If (dict) predict_features is given, predict.
        if graph == False:
            lr_model = LinearRegression()
            lr_model.fit(self.feature_train, self.label_train)
            label_predict = lr_model.predict(self.feature_test)
            test_acc = lr_model.score(self.feature_test, self.label_test)

            coef = lr_model.coef_
            for i in coef:
                coefficient = i
            inter = lr_model.intercept_
            for i in inter:
                intercept = i

        if graph == True:
            lr_model = LinearRegression()
            lr_model.fit(self.feature_train, self.label_train)
            label_predict = lr_model.predict(self.feature_test)
            test_acc = lr_model.score(self.feature_test, self.label_test)
            
            coef = lr_model.coef_
            for i in coef:
                coefficient = i
            inter = lr_model.intercept_
            for i in inter:
                intercept = i

            feature_title = [i for i in self.feature_test.columns]
            label_title = [i for i in self.label_test.columns]

            sns.regplot(x=self.label_test, y=label_predict, ci=None, color="r")

            plt.scatter(self.label_test, label_predict, alpha=0.3)
            plt.xlabel(f"{feature_title}")
            plt.ylabel(f"{label_title}")
            plt.title(title)
            plt.show()

            # Correlation
            correlation = df.corr()
            sns.set(rc={"figure.figsize": (15, 10)})
            sns.heatmap(correlation, cmap="seismic", annot=True, vmin=-1, vmax=1)
            plt.show()

        # Prediction
        if predict_features != {}:
            features_used = list(self.feature_train.columns)
            predict_features_lst = []
            for key, val in predict_features.items():
                if key in features_used:
                    predict_features_lst.append(val)
                
            prediction = lr_model.predict([predict_features_lst])
            for i in prediction:
                for j in i:
                    prediction = j
            print(f"Test Accuracy : {test_acc}\nPredicted Label : {prediction}\nCoefficients : {coefficient}\nIntercept : {intercept}")
            return prediction
        else:
            print(f"Test Accuracy : {test_acc}\nCoefficients : {coefficient}\nIntercept : {intercept}")
            


    #def k_neighbors_classifier(self, ):

    #def neural_network(self, ):
    pass


class FeatureOptimizer:
    def __init__(self, feature_train, feature_test, label_train, label_test):
        self.feature_train = feature_train
        self.feature_test = feature_test
        self.label_train = label_train
        self.label_test = label_test
    
    def optimized_feature_lr(self):
        feature_optim = []
        score_lst = []
        index_lst = []
        group_of_features = []

        for n in range(len(self.feature_train.columns)):
            iter_n = itertools.combinations(range(0, len(self.feature_train.columns)), n+1)
            index_lst.append(list(iter_n))
        
        for i in index_lst:
            for j in i:
                sub_lst = []
                for ind in j:
                    find_feature = list(self.feature_train.columns)[ind]
                    sub_lst.append(find_feature)
                group_of_features.append(sub_lst)
        
        for features in group_of_features:
            feature_train = self.feature_train[features]
            feature_test = self.feature_test[features]

            lr_model = LinearRegression()
            lr_model.fit(feature_train, self.label_train)
            test_acc = lr_model.score(feature_test, self.label_test)
            score_lst.append(test_acc)
            feature_optim.append(features)
        
        max_score = max(score_lst)
        max_index = score_lst.index(max_score)
        optimum_feature = feature_optim[max_index]

        feature_train_op = self.feature_train[optimum_feature]
        feature_test_op = self.feature_test[optimum_feature]
        return feature_train_op, feature_test_op

class OptimumAlgorithmSearch:
    pass