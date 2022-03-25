import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict


def booleator(col):
    if str(col).lower() in ['true', 'yes']:
        return True
    else:
        return False


def draw_throughput_plot(tpx, tpy, label):
    plt.scatter(tpx, tpy)
    plt.xscale("log")
    plt.xlabel("message size")
    plt.ylabel("Mbit/s")
    plt.title(label)
    plt.show()


def load_df(path):
    return  pd.read_csv(path, sep='\s*,\s*', converters={'roughness': booleator, 'unstab': booleator},
                               engine='python')


if __name__ == "__main__":
    tp1n_dataframe = load_df('tp1n2.txt')
    tpmn_dataframe = load_df('tpmn2.txt')

    draw_throughput_plot(tp1n_dataframe['msg size'], tp1n_dataframe['tp [mbit/s]'], 'single node [2p per n]')
    draw_throughput_plot(tpmn_dataframe['msg size'], tpmn_dataframe['tp [mbit/s]'], 'multi node [1p per n]')

    lat1n_dataframe = load_df('lat1n2.txt')
    latmn_dataframe = load_df('latmn2.txt')

    print(f'single node [2p per n]: {lat1n_dataframe["time [ns]"][0]/1e6}ms')
    print(f'multi node [1p per n]: {latmn_dataframe["time [ns]"][0]/1e6}ms')