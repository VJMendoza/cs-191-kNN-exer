import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import getopt
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

DATA_PATH = '../Data/'
HABERMAN_DATA = 'haberman.data'

NAMES = ['Age of Patient', 'Year of Operation',
         'Axillary nodes detected', 'Survival Status']
FEATURE_COLUMNS = NAMES[:3]
Y_COLUMN = [NAMES[3]]


def test():
    return 'this is a test'


def main(argv):
    neighbors = 0
    try:
        opts, _ = getopt.getopt(
            argv, 'hk:', ['help', 'neighbors'])
    except getopt.GetoptError:
        print('Invalid argument')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('pca_face.py -k <k_neighboors>')
        elif opt in ('-k', '--neighbors'):
            neighbors = arg

    return neighbors


def read_csv(file_path=os.path.join(DATA_PATH, HABERMAN_DATA)):
    dataset = pd.read_csv(file_path, names=NAMES)
    return dataset


def split_dataset(x_values, y_values):
    return train_test_split(x_values, y_values,
                            test_size=0.2, random_state=191)


def create_classifier(x_train, y_train, x_test, y_test, k=3):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    return confusion_matrix(y_test, pred)


def use_diff_k(x_train, y_train):
    k_list = list(range(1, 25))
    cv_scores = []

    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(
            knn, x_train, y_train, cv=10, scoring='accuracy')
        print('At {0}: score is {1}'.format(k, scores.mean()))
        cv_scores.append(scores.mean())

    acc = [x for x in cv_scores]
    return k_list, acc


def plot_results(k_list, acc):
    plt.figure(figsize=(15, 10))
    plt.title('The optimal number of neighbors',
              fontsize=20, fontweight='bold')
    plt.xlabel('Number of Neighbors K', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    sns.set_style("whitegrid")
    plt.plot(k_list, acc)
    plt.show()

    best_k = k_list[mse.index(max(acc))]
    print('Optimal number of neighbors is {0}'.format(best_k))


if __name__ == "__main__":
    neighbors = int(main(sys.argv[1:]))

    dataset = read_csv()
    print('Dataset loaded.')

    x_values = dataset[FEATURE_COLUMNS].values
    y_values = dataset[Y_COLUMN].values.ravel()

    x_train, x_test, y_train, y_test = split_dataset(x_values, y_values)
    print('Data split.')

    k_list, mse = use_diff_k(x_train, y_train)

    plot_results(k_list, mse)
