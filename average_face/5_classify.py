import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def selector(f, x, y, i):
    model = SelectKBest(score_func=f_classif, k=i)
    model.fit(x, y)
    select_best_col = model.get_support(True)
    print(select_best_col)
    F = f.iloc[:, select_best_col]
    return F


dataset_path = 'data/data_657.csv'
dataset = pd.read_csv(dataset_path)
# N: 155, E: 141, O: 22, A: 81, C: 258
print(Counter(list(dataset['T'])))

# Sampling separately
# dataset_1 = dataset[(dataset['T'] == 1)]  # N:155
# dataset_2 = dataset[(dataset['T'] == 2)]  # E:141
# dataset_3 = dataset[(dataset['T'] == 3)]  # O:22
# dataset_4 = dataset[(dataset['T'] == 4)]  # A:81
# dataset_5 = dataset[(dataset['T'] == 5)]  # C:258
# sample_1 = dataset_1.sample(n=100, axis=0)
# sample_2 = dataset_2.sample(n=100, axis=0)
# sample_5 = dataset_5.sample(n=100, axis=0)
# dataset_381 = pd.concat([sample_1, sample_2, dataset_4, sample_5], axis=0).sort_index()

features = dataset.iloc[:, -25:]
col = features.shape[1]
labels = dataset['T']
X = features.values
Y = labels.values

for step in range(5, col+1):
    X0 = selector(features, X, Y, step)
    y0 = Y
    X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accurate = accuracy_score(y_test, y_pred)
    print('accurate:', '%.2f' % accurate)
