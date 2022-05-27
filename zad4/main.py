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

def load_data():
    return pd.read_csv('out.csv', sep='\s*,\s*',
                       converters={'roughness': booleator, 'unstab': booleator},
                       engine='python')

def unique_values(df, fields):
    return df[fields].drop_duplicates()

def filter_on(df, fields, value):
    return df[df[fields].apply(tuple, 1) == tuple(value)]

def iterate_over_field_values_impl(df, fields, values, func):
    if len(fields) == 0:
        func(df, values)
    else:
        unique_field_values = unique_values(df, fields[0])
        for _, unique_field_value in unique_field_values.iterrows():
            filtered = filter_on(df, fields[0], unique_field_value)
            iterate_over_field_values_impl(filtered, fields[1:], values + [tuple(unique_field_value)], func)

def iterate_over_field_values(df, fields, func):
    iterate_over_field_values_impl(df, fields, [], func)

def single_thread_values(df, mean_str):
    return float(df[df['num_threads'] == 1][mean_str])

def time_taken_series(df, num_threads, mean_str, std_str):
    x_vs = num_threads
    y_vs = list(df[mean_str])
    y_errs = list(df[std_str])

    return x_vs, y_vs, y_errs

def speedup_series(df, num_threads, mean_str, std_str):
    x_vs, y_vs, y_errs = time_taken_series(df, num_threads, mean_str, std_str)

    y_single = single_thread_values(df, mean_str)

    y_speedups = [y_single / y_v if y_v > 0.0 else 0.0 for y_v in y_vs]
    y_speedup_errs = [y_single / y_err if y_err > 0.0 else 0.0 for y_err in y_errs]

    return x_vs, y_speedups, y_speedup_errs

def efficency_series(df, num_threads, mean_str, std_str):
    x_vs, y_vs, y_errs = speedup_series(df, num_threads, mean_str, std_str)

    y_efficencies = [y_v/x_v for x_v, y_v in zip(x_vs, y_vs)] if len(x_vs) != 0 else [0]
    y_efficency_errs = [y_err/x_v for x_v, y_err in zip(x_vs, y_errs)] if len(x_vs) != 0 else [0]

    return x_vs, y_efficencies, y_efficency_errs

def make_time_taken_subplot(df, num_threads, plt0, mean_str, std_str, label):
    x, y, y_err = time_taken_series(df, num_threads, mean_str, std_str)
    if len(x) != len(y):
        return
    plt0.errorbar(x, y, yerr=y_err, label=label)
    plt0.set_ylim([0, max(1, max(y)*1.1, plt0.get_ylim()[1])])

def make_speedup_subplot(df, num_threads, plt0, mean_str, std_str, label):
    x, y, y_err = speedup_series(df, num_threads, mean_str, std_str)
    if len(x) != len(y):
        return
    plt0.plot(x, y, label=label)

def make_efficiency_subplot(df, num_threads, plt0, mean_str, std_str, label):
    x, y, y_err = efficency_series(df, num_threads, mean_str, std_str)
    if len(x) != len(y):
        return
    plt0.plot(x, y, label=label)

def set_time_taken_subplot_properties(plt0, title, num_threads):
    plt0.set_title(title)
    plt0.set_ylabel('time taken [ms]')
    plt0.set_xlabel('thread count')
    plt0.set_xscale('log')
    plt0.set_xticks(num_threads)
    plt0.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

def set_speedup_subplot_properties(plt0, title, num_threads):
    plt0.set_title(title)
    plt0.set_ylabel('speedup')
    plt0.set_xlabel('thread count')
    plt0.set_xscale('log')
    plt0.set_xticks(num_threads)
    plt0.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt0.set_ylim([0, 12.25])

def set_efficiency_subplot_properties(plt0, title, num_threads):
    plt0.set_title(title)
    plt0.set_ylabel('efficiency')
    plt0.set_xlabel('thread count')
    plt0.set_xscale('log')
    plt0.set_xscale('log')
    plt0.set_xticks(num_threads)
    plt0.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt0.set_ylim([0, 1.25])

def ideal_none(plt0, num_threads):
    pass

def ideal_speedup(plt0, num_threads):
    xs = list(num_threads)
    ys = [v for v in xs]
    plt0.plot(xs, ys, '--')

def ideal_efficiency(plt0, num_threads):
    xs = list(num_threads)
    ys = [1 for _ in xs]
    plt0.plot(xs, ys, '--')

def make_measurement_figure(df, vs, num_threads, subplot_func, subplot_props_func, ideal_func):
    length, desired_bucket_size, final_bucket_size_coeff = vs[0][0], vs[1][0], vs[2][0]
    allocation_preset, total_length_schedule, buckets_count_schedule = vs[3][0], vs[4][0], vs[5][0]

    fig = plt.figure(figsize=(14,12), dpi=80)

    allocation_plt = fig.add_subplot(421)
    subplot_props_func(allocation_plt, 'allocation', num_threads)
    generation_plt = fig.add_subplot(422)
    subplot_props_func(generation_plt, 'generation', num_threads)
    bucketization_plt = fig.add_subplot(423)
    subplot_props_func(bucketization_plt, 'bucketization', num_threads)
    sequential_sorting_plt = fig.add_subplot(424)
    subplot_props_func(sequential_sorting_plt, 'sequential sorting', num_threads)
    writing_back_plt = fig.add_subplot(425)
    subplot_props_func(writing_back_plt, 'writing back', num_threads)
    concatenation_plt = fig.add_subplot(426)
    subplot_props_func(concatenation_plt, 'concatenation', num_threads)
    sort_total_plt = fig.add_subplot(427)
    subplot_props_func(sort_total_plt, 'sorting total', num_threads)
    total_plt = fig.add_subplot(428)
    subplot_props_func(total_plt, 'total', num_threads)

    fields = ['algorithm version', 'bucket type']
    version_bucket_types = unique_values(df, fields)
    for _, vbt in version_bucket_types.iterrows():
        filtered = filter_on(df, fields, vbt)
        label = f'version: {vbt["algorithm version"]} + bucket type: {vbt["bucket type"]}'
        subplot_func(filtered, num_threads, allocation_plt, 'time allocation (mean)', 'time allocation (std)', label)
        subplot_func(filtered, num_threads, generation_plt, 'time generation (mean)', 'time generation (std)', label)
        subplot_func(filtered, num_threads, bucketization_plt, 'time bucketization (mean)', 'time bucketization (std)', label)
        subplot_func(filtered, num_threads, sequential_sorting_plt, 'time sequential sorting (mean)', 'time sequential sorting (std)', label)
        subplot_func(filtered, num_threads, writing_back_plt, 'time writing back (mean)', 'time writing back (std)', label)
        subplot_func(filtered, num_threads, concatenation_plt, 'time concatenation (mean)', 'time concatenation (std)', label)
        subplot_func(filtered, num_threads, sort_total_plt, 'time total sort (mean)', 'time total sort (std)', label)
        subplot_func(filtered, num_threads, total_plt, 'time total (mean)', 'time total (std)', label)

    ideal_func(allocation_plt, num_threads)
    ideal_func(generation_plt, num_threads)
    ideal_func(bucketization_plt, num_threads)
    ideal_func(sequential_sorting_plt, num_threads)
    ideal_func(writing_back_plt, num_threads)
    ideal_func(concatenation_plt, num_threads)
    ideal_func(sort_total_plt, num_threads)
    ideal_func(total_plt, num_threads)

    fig.subplots_adjust(hspace=.5, wspace=0.5)
    handles, labels = allocation_plt.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f'length={length},'
              f' desired bucket size={desired_bucket_size},\n'
              f' final bucket size coeff={final_bucket_size_coeff},'
              f' total length schedule={total_length_schedule},\n'
              f' buckets count schedule={buckets_count_schedule},'
              f' allocation preset={allocation_preset}')

    return fig

def total(df, fields):
    accum = 1
    for field in fields:
        uvs = unique_values(df, field)
        accum *= len(uvs.index)

    return accum

import gc

if __name__ == "__main__":
    unfiltered = load_data()

    num_threads = unfiltered['num_threads'].unique()

    fields = [
        ['length'],
        ['desired bucket size'],
        ['final bucket size coeff'],
        ['allocation preset'],
        ['total length schedule'],
        ['buckets count schedule'],
    ]

    count = total(unfiltered, fields)
    global i
    i = 0

    def draw_fig(df, vs):
        global i
        time_taken = make_measurement_figure(df, vs, num_threads, make_time_taken_subplot, set_time_taken_subplot_properties, ideal_none)
        speedup = make_measurement_figure(df, vs, num_threads, make_speedup_subplot, set_speedup_subplot_properties, ideal_speedup)
        efficiency = make_measurement_figure(df, vs, num_threads, make_efficiency_subplot, set_efficiency_subplot_properties, ideal_efficiency)
        time_taken.savefig(f'out_graphs/{i}_time_taken.png')
        speedup.savefig(f'out_graphs/{i}_speedup.png')
        efficiency.savefig(f'out_graphs/{i}_efficiency.png')
        i+=1
        print(f'progress {i/count*100.0}%')
        for ax in time_taken.get_axes():
            time_taken.delaxes(ax)
        for ax in speedup.get_axes():
            speedup.delaxes(ax)
        for ax in efficiency.get_axes():
            efficiency.delaxes(ax)
        time_taken.clf()
        time_taken.clear()
        speedup.clf()
        speedup.clear()
        efficiency.clf()
        efficiency.clear()
        plt.close(time_taken)
        plt.close(speedup)
        plt.close(efficiency)
        del time_taken
        del speedup
        del efficiency
        plt.close('all')
        plt.clf()
        plt.cla()
        gc.collect()

    iterate_over_field_values(unfiltered, fields, draw_fig)
