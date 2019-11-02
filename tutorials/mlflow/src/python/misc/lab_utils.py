import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import os
import numpy as np

class Utils:

    @staticmethod
    def load_data(path):
        """
        Read a CSV file from a given path and return a Pandas DataFrame
        :param path: path to csv file
        :return: returns Pandas DataFrame
        """

        df = pd.read_csv(path)
        return df

    @staticmethod
    def plot_graphs(x_data, y_data, x_label, y_label, title):
        """
        Use the Mathplot lib to plot data points provide and respective x-axis and y-axis labels
        :param x_data: Data for x-axis
        :param y_data: Data for y-axis
        :param x_label: Label for x-axis
        :param y_label: LabEL FOR Y-axis
        :param title: Title for the plot
        :return: return a tuple (fig, ax)
        """

        plt.clf()

        fig, ax = plt.subplots()
        ax.plot(x_data, y_data)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        return (fig, ax)

    @staticmethod
    def plot_residual_graphs(predictions, y_test, x_label, y_label, title):
        """
        Create residual plot using seaborn plotting library
        https://seaborn.pydata.org/tutorial/regression.html
        :param predictions: predictions from the run
        :param y_test: actual labels
        :param x_label: name for the x-axis
        :param y_label: name for the y-axis
        :param title:  title for the plot
        :return: tuple of plt, fig, ax
        """

        fig, ax = plt.subplots()

        sns.residplot(predictions, y_test, lowess=True)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        return (plt, fig, ax)

    @staticmethod
    def get_mlflow_directory_path(*paths, create_dir=True):
        """
        Get the current running path where mlruns is created. This is the directory from which
        the python file containing MLflow code is executed. This method is used for artifacts, such
        as images, where we want to store plots.
        :param paths: list of directories below mlfruns, experimentID, mlflow_run_id
        :param create_dir: detfault is True
        :return: path to directory.
        """

        cwd = os.getcwd()
        dir = os.path.join(cwd, "mlruns", *paths)
        if create_dir:
            if not os.path.exists(dir):
                os.mkdir(dir, mode=0o755)
        return dir

    @staticmethod
    def get_temporary_directory_path(prefix, suffix):
        """
        Get a temporary directory and files for artifacts
        :param prefix: name of the file
        :param suffix: .csv, .txt, .png etc
        :return: object to tempfile.
        """

        temp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix)
        return temp

    @staticmethod
    def print_pandas_dataset(d):
        """
        Given a Pandas dataFrame show the dimensions sizes
        :param d: Pandas dataFrame
        :return: None
        """
        print("rows = %d; columns=%d" % (d.shape[0], d.shape[1]))
        print(d.head())

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Borrowed from the scikit-learn library documentation

        :param y_true: the actual value of y
        :param y_pred: the predicted valuye of y
        :param classes: list of label classes to be predicted
        :param normalize: normalize the data
        :param title: title of the plot for confusion matrix
        :param cmap: color of plot
        :return: returns a tuple of (plt, fig, ax)
        """

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return (plt, fig, ax)
