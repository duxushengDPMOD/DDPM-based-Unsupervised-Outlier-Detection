import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import argparse
def show_result_test(x_0, Original_data):

    l1 = np.zeros(x_0.shape[0])
    l2 = np.ones(x_0.shape[0])
    datas = np.concatenate((Original_data, x_0), axis=0)
    labels = np.concatenate((l1, l2), axis=0)

    draw_plot_test(datas, labels, 2, 20150101)

def draw_plot_test(data, label, dimensions, rs):
    X_norm = data
    normal_idxs = (label == 0)
    abnorm_idxs = (label == 1)
    X_original = X_norm[normal_idxs]

    X_generate = X_norm[abnorm_idxs]

    plt.figure(figsize=(9, 9), dpi=300)
    plt.scatter(X_original[:, 0], X_original[:, 1], 20, color='lightsteelblue')
    plt.scatter(X_generate[:, 0], X_generate[:, 1], 20, color='lightcoral')


    plt.title("Restored Result")

    plt.show()

def show_allObject_loss(loss_list):


    epochs = range(1, len(loss_list) + 1)


    plt.plot(epochs, loss_list, marker='o')

    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


    plt.show()

def show_PCA_Results(x_0,original_x):

    pca1 = PCA(n_components=2)
    pca1.fit(x_0)

    reduced_data_x_0 = pca1.transform(x_0)
    reduced_data_x_0=reduced_data_x_0[:, :2]


    pca2 = PCA(n_components=2)
    pca2.fit(original_x)

    reduced_data_original_x = pca2.transform(original_x)
    reduced_data_original_x = reduced_data_original_x[:, :2]



    fig,ax=pl.subplots()
    ax.scatter(reduced_data_x_0[:, 0], reduced_data_x_0[:, 1],label='X_0')
    ax.scatter(reduced_data_original_x[:, 0], reduced_data_original_x[:, 1], label='Original_data')

    ax.legend()
    plt.title("PCA Result")
    plt.show()
