from PyML import DataProcessing as dp
from PyML import MachineLearning as ml
from PyML import FeatureOptimizer as op
'''
pydp = dp("./test.csv")
df = pydp.csv_to_DataFrame(True)
#print(f"DATAFRAME\n{df}")

feature, label = pydp.feature_label_split(df, ["f1", "f2", "f3"], ["label"])
#print(f"\nFEATURE: {feature}\nLABEL: {label}")

feature_train, feature_test, label_train, label_test = pydp.train_test_split(feature, label)
print(feature_train)

lst = pydp.df_to_dict(df)
#print(lst)

dummies = pydp.to_dummies(df, ["f2"])
#print(dummies)
'''


pydp = dp("./weather.csv")

df = pydp.csv_to_DataFrame(True)

feature_col = ["avg_relative_humidity", "avg_wind_speed", "avg_pressure_sea", "avg_pressure_station", "precipitation", "rain", "snow"]
label_col = ["avg_temperature"]
test_data = [86.5, 7,101.63,99.49,0,0,0]
feature_dict = dict(zip(feature_col, test_data))

df = pydp.to_floats(df, feature_col)
feature, label = pydp.feature_label_split(df, feature_col, label_col)
feature_train, feature_test, label_train, label_test = pydp.train_test_split(feature, label)

pyop = op(feature_train, feature_test, label_train, label_test)
feature_train_op, feature_test_op = pyop.optimized_feature_lr()

pyml = ml(feature_train, feature_test, label_train, label_test)
model = pyml.linear_regression(df, graph=False, title="Predict Temperature", predict_features=feature_dict)
print(model)

pyml1 = ml(feature_train_op, feature_test_op, label_train, label_test)
model1 = pyml1.linear_regression(df, graph=False, title="Predict Temperature", predict_features=feature_dict)
print(model1)


