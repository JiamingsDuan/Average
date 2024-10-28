# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
N : Counter({2.0: 304, 3.0: 256, 1.0: 97})
E : Counter({2.0: 277, 3.0: 275, 1.0: 105})
O : Counter({2.0: 315, 3.0: 253, 1.0: 89})
A : Counter({3.0: 304, 2.0: 272, 1.0: 81})
C : Counter({2.0: 290, 3.0: 269, 1.0: 98})
'''

feature_path = 'data/feature_25.csv'
Features = pd.read_csv(feature_path)
label_path = 'data/label_657.csv'
Labels = pd.read_csv(label_path)[['N', 'E', 'O', 'A', 'C']]


def generate_train_test(f, la, t):
    X1 = f[f[t] == 1].iloc[:, -25:]
    y1 = la[la[t] == 1][t]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)
    X2 = f[f[t] == 2].iloc[:, -25:]
    y2 = la[la[t] == 2][t]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)
    X3 = f[f[t] == 3].iloc[:, -25:]
    y3 = la[la[t] == 3][t]
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=0)
    X0_train = pd.concat([X1_train, X2_train, X3_train], axis=0)
    X0_test = pd.concat([X1_test, X2_test, X3_test], axis=0)
    y0_train = pd.concat([y1_train, y2_train, y3_train], axis=0)
    y0_test = pd.concat([y1_test, y2_test, y3_test], axis=0)
    train_set = pd.concat([X0_train, y0_train], axis=1).sample(frac=1.0)
    test_set = pd.concat([X0_test, y0_test], axis=1).sample(frac=1.0)

    return train_set, test_set


# Iso type sampling
for i in Labels.columns:
    Features.insert(1, i, value=Labels[i])
    dataset_1 = Features[(Features[i] == 1)]
    dataset_2 = Features[(Features[i] == 2)]
    dataset_3 = Features[(Features[i] == 3)]
    sample_1 = dataset_1.sample(n=80, axis=0)
    sample_2 = dataset_2.sample(n=80, axis=0)
    sample_3 = dataset_3.sample(n=80, axis=0)
    Balance_features = pd.concat([sample_1, sample_2, sample_3], axis=0).sort_index()
    feature = Balance_features.iloc[:, -25:]
    label = Balance_features[i]
    col = feature.shape[1]
    X = feature.values
    y = label.values
    for step in range(5, col+1):
        selector = SelectKBest(score_func=f_classif, k=step)
        selector.fit(X, y)
        select_best_index = selector.get_support(True)
        X0 = feature.iloc[:, select_best_index].values
        y0 = y
        X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accurate = accuracy_score(y_test, y_pred)
        print(i, ':', 'accurate:', '%.2f' % accurate, 'feature_num:k=', step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accurate = accuracy_score(y_test, y_pred)
    print(i, ':', 'accurate:', '%.2f' % accurate)

# Equal proportional division
for j in Labels.columns:
    Features.insert(1, j, value=Labels[j])
    train, test = generate_train_test(f=Features, la=Labels, t=j)
    train_shuffled = shuffle(train)
    test_shuffled = shuffle(test)
    M_train = train.iloc[:, :-1].values
    M_test = test.iloc[:, :-1].values
    n_train = train.iloc[:, -1:].values
    n_test = test.iloc[:, -1:].values
    sc = StandardScaler()
    M_train = sc.fit_transform(M_train)
    M_test = sc.transform(M_test)
    classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    classifier.fit(M_train, n_train.ravel())
    n_pred = classifier.predict(M_test)
    accurate = accuracy_score(n_test, n_pred)
    print(j, ':', 'accurate:', '%.2f' % accurate)
