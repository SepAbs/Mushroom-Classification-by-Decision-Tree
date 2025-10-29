from pandas import read_csv, get_dummies # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text #, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from matplotlib.pyplot import subplots, title, fill_between, xlabel, ylabel, figure, show
from seaborn import histplot, heatmap
from time import time
"""
from IPython.display import Image
from pydotplus import graph_from_dot_data
from graphviz import Source
"""

def Modeling(Model):
    startTime = time()
    # Obtaining best paramteres by using splitted validation data
    Model.fit(X_val, y_val)
    Duration = time() - startTime # Calculating how long fitting train data takes

    print(f"\n{Model.best_score_}\n\n{Model.best_params_}")
    Model = Model.best_estimator_ # = DecisionTreeClassifier(Optimal Parameters)
    print(f"\n{Model}")

    Model.fit(X_train, y_train)
    y_pred = Model.predict(X_test)
    print(f"\nModel Score: {Model.score(X_test, y_pred)}\n\nAccuracy Score: {accuracy_score(y_test, y_pred)}\n\nPrecision Score: {precision_score(y_test, y_pred)}\n\nRecall Score: {recall_score(y_test, y_pred)}\n\nF1-Score: {f1_score(y_test,y_pred)}\n\nComputation Time: {Duration}")

    # Confusion matrix for train set
    _pred = Model.predict(X_train)
    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_train, _pred), display_labels = ["True", "False"]).plot()
    show()

    # Confusion matrix for validation set
    _pred = Model.predict(X_val)
    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_val, _pred), display_labels = ["True", "False"]).plot()
    show()

    # Confusion matrix for test set
    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = ["True", "False"]).plot()
    show()

    print("Decision Tree Classifier report: \n\n", classification_report(y_test, y_pred))

    # Texual representation of modelled decision tree
    print(export_text(Model))

    """
    figure(figsize=(12,8))
    plot_tree(Model, feature_names = df[x_vars].columns, max_depth = 3, filled = True)
    Source(export_graphviz(Model, out_file = None, class_names = ["Edible", "Poisonous"], filled = True, rounded = True), format = "png") 
    Image(graph_from_dot_data(export_graphviz(Model, out_file = None, class_names = ["Edible", "Poisonous"], filled = True, rounded = True)).create_png())
    """

# Open & read dataset
df = read_csv("mushrooms.csv").copy()

# Decribing raw data records
nSamples, nFeatures = df.shape
print(f"Data record has {nSamples} samples and {nFeatures} features.\n")
df.describe()
df.info()
df.isnull().values.any() # Check if any missing value exists.
print(df.shape)

# Plotting raw data records
Figure, ax = subplots()
ax.pie(df["class"].value_counts(), labels = ["Edible", "Poisonous"], autopct = '%1.1f%%', shadow = True, startangle = 90)
ax.axis("equal")
show()
histplot(df["class"])
show()
"""
for Index, Feature in enumerate(df.columns[1:]):
    figure(Index)
    countplot(df[Feature]).set_title('{}'.format(Feature))
"""

# Preprocessing data by means of Label Enconding & One Hot Encoding
labelEncoder, Features = LabelEncoder(), df.columns
X_train, X_test, y_train, y_test = train_test_split(get_dummies(df.drop(['class'], axis = 1)), labelEncoder.fit_transform(df['class']), test_size = 0.3, random_state = 42) # 70:30
print(labelEncoder.fit_transform(df['class']).head())
# second, split into X_val, X_test, y_val, y_test
# 'test_size=0.5' split into 50% and 50%. The original data set is 30%; so, it will split into 15% equally.
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42) # 70:15:15

# splitted dataset dimensions
print("\nDimension of splitted datasets are listed below:\nTrain sets: {X_train.shape} {y_train.shape}\n\nValidation sets: {X_val.shape} {y_val.shape}\n\nTest sets: {X_test.shape} {y_test.shape}")

# Train model by Grid Search hyper parameterization approach
print("\nGridSearchCV:")
Modeling(GridSearchCV(estimator = DecisionTreeClassifier(random_state = 42), param_grid = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}, verbose = 1, cv = 3, scoring = "accuracy", error_score= "raise"))

# Train model by Randomized Search hyper parameterization approach
print("\nRandomizedSearchCV:")
Modeling(RandomizedSearchCV(DecisionTreeClassifier(), {"criterion": ["gini", "entropy"], "max_features": ["sqrt", "log2"], "min_samples_leaf": range(1, 100, 1), "max_depth": range(1, 50, 1)}, cv = 10, scoring = "accuracy", n_iter = 20, random_state = 5))

# Label Enconding Only
for Feature in Features:
    df[Feature] = labelEncoder.fit_transform(df[Feature])

# Convert data to have zero mean and unit variance
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(df.iloc[:,1:18]), df.iloc[:, 0], test_size=0.3, random_state = 42) # 70:30

# 'test_size=0.5' split into 50% and 50%. The original data set is 30%; so, it will split into 15% equally.
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42) # 70:15:15

# Train model by Grid Search hyper parameterization approach
print("\nGridSearchCV:")
Modeling(GridSearchCV(estimator = DecisionTreeClassifier(random_state = 42), param_grid = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}, verbose = 1, cv = 3, scoring = "accuracy", error_score= "raise"))

# Train model by Randomized Search hyper parameterization approach
print("\nRandomizedSearchCV:")
Modeling(RandomizedSearchCV(DecisionTreeClassifier(), {"criterion": ["gini", "entropy"], "max_features": ["sqrt", "log2"], "min_samples_leaf": range(1, 100, 1), "max_depth": range(1, 50, 1)}, cv = 10, scoring = "accuracy", n_iter = 20, random_state = 5))

print("\nTHE-END")
