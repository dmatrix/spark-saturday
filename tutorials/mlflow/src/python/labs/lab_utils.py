
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def plot_graphs(x_data, y_data, x_label, y_label):
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt

def get_mlflow_directory_path(*paths, create_dir=True):
    cwd = os.getcwd()
    dir = os.path.join(cwd, "mlruns", *paths)
    if create_dir:
        if not os.path.exists(dir):
            os.mkdir(dir, mode=0o755)
    return dir


def print_pandas_dataset(d):
    print("rows = %d; columns=%d" % (d.shape[0], d.shape[1]))
    print(d.head())
