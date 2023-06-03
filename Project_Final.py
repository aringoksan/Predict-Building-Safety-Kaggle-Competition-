# Written in Python 3.11
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, make_scorer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBRegressor
import warnings
from lightgbm import LGBMRegressor
import time
from sklearn.svm import SVR
import itertools


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
standard_scaler = StandardScaler()
df0 = pd.read_csv("train.csv")
df = df0.copy()


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "int32", "int64", "float32",
                                                                                   "float64"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if
                dataframe[col].dtypes in ["int", "int32", "int64", "float32", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
excel_graph = df[num_cols].describe().T

# Check the correlation of variables
crr = df.corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(crr, cmap="RdBu")
plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

    print("#####################################")

def high_correlated_cols(dataframe, plot=False, corr_th=0.9):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(df, plot=False)

#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


cols_have_outlier = []
for col in num_cols:
    if check_outlier(df, col):
        cols_have_outlier.append(col)
    print(col, "---", check_outlier(df, col))


cols_have_outlier.remove('Max drift mm')
is_missing = df.isnull().sum()

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def feature_engineering(input_name):
    df = pd.read_csv(input_name)
    df = df.drop("soil_class__C", axis=1)
    df["Instensity of the seismic load"] = df["PGA g"] * df["Magnitude"]
    df["earthquacke_size"] = ["low" if value < 6.3 else "medium" if value < 6.6 else "high" for value in
                              df["Magnitude"]]
    df["building_size"] = ["short" if value <= 2 else "medium" if value <= 4 else "tall" for value in
                           df["Number of floors"]]
    df["eq_scale"] = df["Magnitude"] / df["Distance to fault km"]
    df["Non-linear Period s"] = df["Period s"] ** 2
    df["Period s"] = np.log10(df["Period s"])
    df["Floor mass kg"] = np.log10(df["Floor mass kg"])
    df["eq_speed"] = df["Magnitude"] * df["Period s"]
    df["Columns Area Change"] = [0 if df["Columns 4-6 A mm2"][i] == 0
                                 else df["Columns 1-3 A mm2"][i] / df["Columns 4-6 A mm2"][i]
                                 for i in range(len(df["Columns 4-6 A mm2"]))]
    df = one_hot_encoder(df, ["earthquacke_size", "building_size"], drop_first=True)
    return df

def local_outliers(plottable=False):
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(df)

    df_scores = clf.negative_outlier_factor_
    scores = pd.DataFrame(np.sort(df_scores))
    if plottable:
        scores.plot(stacked=True, xlim=[0, 50], style='.-')
        plt.show()

def feature_importance_variables(X_train_a_fe, y_train_a_fe, graph=False):
    rf = RandomForestRegressor().fit(X=X_train_a_fe, y=y_train_a_fe)
    feature_names = X_train_a_fe.columns
    importance = rf.feature_importances_
    sorted_indices = sorted(range(len(importance)), key=lambda k: importance[k], reverse=True)
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importance = [importance[i] for i in sorted_indices]
    if graph:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_importance)), sorted_importance)
        plt.xticks(range(len(sorted_importance)), sorted_feature_names, rotation='vertical')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    return sorted_feature_names


# Target varible distribution plot
df["Max drift mm"].hist(bins=50, width=5, grid=True)
plt.xlabel("Max Drift [mm]", fontsize=18)
plt.ylabel("Value Counts", fontsize=18)
plt.xticks(range(0, 210, 25))
plt.xlim([0, 200])
plt.yticks(range(0, 610, 50))
plt.ylim([0, 600])
plt.title("Distribution of Target Variable", fontsize=24)
plt.show()

# Trains and predicts the values according to input model, if selected predictions are written to output file
def predict_get_results(X_train, X_test, y_train, prediction_df, model, name, feature_engineering=False, train=True,
                        output_name="default.csv"):
    start = time.time()
    fitted_model = model.fit(X_train, y_train)
    if train:
        if feature_engineering:
            y_pred = pd.concat([prediction_df,
                                pd.DataFrame(10 ** cross_val_predict(model, X_train, y_train, cv=5, n_jobs=-1),
                                             columns=[name])], axis=1)
        else:
            y_pred = pd.concat([prediction_df, pd.DataFrame(cross_val_predict(model, X_train, y_train, cv=5, n_jobs=-1),
                                                            columns=[name])], axis=1)
        mape = 100 * mean_absolute_percentage_error(y_pred.iloc[:, 0], y_pred[name])
    else:
        if feature_engineering:
            y_pred = pd.concat([pd.DataFrame(10 ** fitted_model.predict(X_test), columns=["Max drift mm"])], axis=1)
            y_pred.index.names = ['Index']
            y_pred.to_csv(output_name)
            print(output_name + "  file has been generated")
        else:
            y_pred = pd.concat([pd.DataFrame(fitted_model.predict(X_test), columns=["Max drift mm"])], axis=1)
            y_pred.index.names = ['Index']
            y_pred.to_csv(output_name)
            print(output_name + "  file has been generated")
        mape = 0
    end = time.time()
    time_elapsed = end - start
    return y_pred, mape, time_elapsed


# Different models are tested, for quickness all models are commented only random forest is applied

models = {"LinearRegression": LinearRegression(), 'Ridge': Ridge(), 'EN': ElasticNet(), 'DT': DecisionTreeRegressor(),
           "RF": RandomForestRegressor(), "XGB": XGBRegressor(), "LGBM": LGBMRegressor(), "SVM": SVR(), "ETR": ExtraTreesRegressor()}



# Determining the number of columns to drop to get the most accurate result
def determine_feature_importance(plottable=False, input_name="train.csv"):
    mapes = []
    num_dropped = []
    for i in range(6, 25):
        data_a_fe = feature_engineering(input_name)
        X = data_a_fe.drop("Max drift mm", axis=1)
        y = np.log10(data_a_fe["Max drift mm"])
        X = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)
        y_p_fi = (10**y).reset_index(drop=True)
        important_varibles = feature_importance_variables(X, y)
        # Plotting the feature importance
        X.drop(important_varibles[i:], axis=1, inplace=True)
        y_p_fi, mape_a_fe, elapsed_a_fe = predict_get_results(X, X, y,
                                                              y_p_fi, RandomForestRegressor(), "RF", feature_engineering=True)
        print("MAPE Score of " + "RF" + " on test set: " + str(mape_a_fe) + " ##Time passed: " + str(elapsed_a_fe))
        mapes.append(mape_a_fe)
        num_dropped.append(30-i)
    # Plotting number of columns dropped vs accuracy to determine the optimal number of columns to drop
    if plottable:
        plt.figure(figsize=(10, 6))
        plt.plot(num_dropped, mapes, color="red", marker="o")
        plt.xlabel('Number of Features Dropped', fontsize=18)
        plt.ylabel('MAPE', fontsize=18)
        plt.xticks(range(5, 27))
        plt.xlim([5, 26])
        plt.yticks(np.arange(12, 16.6, 0.5))
        plt.ylim([12, 16.5])
        plt.tight_layout()
        plt.show()
    return 0

# Use if you want to see the effect of dropping variables on MAPE
# determine_feature_importance(plottable=True)

data_a_fe = feature_engineering("train.csv")
data_test = feature_engineering("test.csv")

X_a_fe = data_a_fe.drop("Max drift mm", axis=1)
y_a_fe = np.log10(data_a_fe["Max drift mm"])
X_a_fe = pd.DataFrame(standard_scaler.fit_transform(X_a_fe), columns=X_a_fe.columns)

# For kaggle output use below
# data_test = pd.DataFrame(standard_scaler.fit_transform(data_test), columns=data_test.columns)

y_pred_a_fe = (10 ** y_a_fe).reset_index(drop=True)
important_varibles = feature_importance_variables(X_a_fe, y_a_fe, graph=True)
X_a_fe.drop(important_varibles[12:], axis=1, inplace=True)
data_test.drop(important_varibles[12:], axis=1, inplace=True)



# Finding different model accuracies after feature engineering
for name, model in models.items():
    y_pred_a_fe, mape_a_fe, elapsed_a_fe = predict_get_results(X_a_fe, X_a_fe, y_a_fe,
                                                               y_pred_a_fe, model, name, feature_engineering=True)
    print("MAPE Score of " + name + " on test set: " + str(mape_a_fe) + " ##Time passed: " + str(elapsed_a_fe))

# For output of kaggle use below command
# y_pred_a_fe, mape_a_fe, elapsed_a_fe = predict_get_results(X_train_a_fe, data_test, y_train_a_fe, y_pred_a_fe,
#                                                            LGBMRegressor(), "LGBM", train=False, output_name="output_V7.csv")


# Hyperparameter optimization, Extra tree regressor model is selected since it was the most accurate base model
grid_params = {"n_estimators": [100, 250], "min_samples_split": [2, 10],
               "max_depth": [1000], "min_samples_leaf": [1, 10]}
min_mape = np.inf
y_pred_a_fe = (10 ** y_a_fe).reset_index(drop=True)
mapes = []
index = 0
hyperparameters = GridSearchCV(ExtraTreesRegressor(), param_grid=grid_params, verbose=0).fit(X_a_fe, y_a_fe)
best_parameters = hyperparameters.best_params_
final_model = hyperparameters
final_pred, final_mape, final_elapsed = predict_get_results(X_a_fe, X_a_fe, y_a_fe, y_pred_a_fe, final_model, "Final Model",
                                                            feature_engineering=True)
print("MAPE Score of " + "Final Model" + " on test set: " + str(final_mape) + " ##Time passed: " + str(final_elapsed))
print("Done")