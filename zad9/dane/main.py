import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def booleator(col):
    if str(col).lower() in ['true', 'yes']:
        return True
    # elif str(col).lower() == "false":
    #    return False
    else:
        return False

def load_data(path: str):
    return pd.read_csv(path, sep='\s*,\s*',
                       converters={'roughness': booleator, 'unstab': booleator},
                       engine='python')


def plot(ds, label):
    plt.plot(ds['size'], ds['time [msec]'], label=label)


if __name__ == "__main__":
    reduction_global = load_data('1reduction_global.txt')
    reduction_shared = load_data('1reduction_shared.txt')
    warp_divergence_sequential = load_data('2warp_divergence_sequential.txt')
    warp_divergence_interleaving = load_data('2warp_divergence_interleaving.txt')
    loop_unrolling_wo_cg = load_data('3loop_unrolling_without_cooperative_grouping.txt')
    loop_unrolling_w_cg = load_data('3loop_unrolling_with_cooperative_grouping.txt')
    atomic_simple = load_data('4atomic_operations_simplest.txt')
    atomic_block = load_data('4atomic_operations_block.txt')
    atomic_warp = load_data('4atomic_operations_warp.txt')

    histo = load_data('5histo.txt')

    plt.figure(figsize=(8, 6), dpi=100)

    plot(reduction_global, '1.global')
    plot(reduction_shared, '1.shared')
    plot(warp_divergence_sequential, '2.warp_div.sequential')
    plot(warp_divergence_interleaving, '2.warp_div.interleaved')
    plot(loop_unrolling_w_cg, '3.unrolling.w_cg')
    plot(loop_unrolling_wo_cg, '3.unrolling.wo_cg')
    plot(atomic_simple, '4.atomic.simple')
    plot(atomic_block, '4.atomic.block')
    plot(atomic_warp, '4.atomic.warp')

    plt.xticks(reduction_global['size'])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('time [ms]')
    plt.xlabel('size')
    plt.legend()
    plt.savefig('reduction_all.png')
    plt.show()

    for bin_c in histo['bin_count'].unique():
        hist_filt = histo[histo['bin_count'] == bin_c]

        plt.figure(figsize=(8, 6), dpi=100)

        plt.plot(hist_filt['size'], hist_filt['naive [ms]'], label='histo.naive')
        plt.plot(hist_filt['size'], hist_filt['simple [ms]'], label='histo.simple')
        plt.xticks(hist_filt['size'])
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('time [ms]')
        plt.xlabel('size')
        plt.legend()
        plt.title(f'bin count={bin_c}')
        plt.savefig(f'histo_bin_{bin_c}.png')
        plt.show()
