import matplotlib.pyplot as plt;
import numpy as np;
import itertools
from collections import defaultdict

def load_formatted(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        return list(map(lambda s: s[:-1], lines))


def parse_tuple(pc_time_tuple: str):
    res = pc_time_tuple.split(sep=',')
    proc_count = int(res[0])
    time_taken = float(res[1])
    return (proc_count, time_taken)


def split_reps(size_data):
    reps = [list(y) for x, y in itertools.groupby(size_data, lambda z: z == 'sep') if not x]
    parsed_reps = list(map(lambda rep: list(map(lambda s: parse_tuple(s), rep)), reps))
    return [{k: v for k, v in rep} for rep in parsed_reps]


def split_data(raw_data):
    s1 = raw_data.index('1000000')
    s2 = raw_data.index('10000000')
    s3 = raw_data.index('1000000000')

    s1_data = raw_data[s1+1:s2]
    s2_data = raw_data[s2+1:s3]
    s3_data = raw_data[s3+1:]

    return {
        1000000: split_reps(s1_data),
        10000000: split_reps(s2_data),
        1000000000: split_reps(s3_data),
    }


def x_for_proc(reps: list[dict[int, float]]):
    time_for_procc = defaultdict(lambda: [])
    for rep in reps:
        for k, v in rep.items():
            time_for_procc[k].append(v)
    return dict(time_for_procc)


def mean_std(arr):
    return np.mean(arr), np.std(arr)


def mean_std_for_proc(time_for_proc):
    return {k: mean_std(v) for k, v in time_for_proc.items()}


def transform_speedup(reps: list[dict[int, float]]):
    return [
        {k: rep[1]/v for k, v in rep.items()} for rep in reps
    ]


def transform_efficiency(reps: list[dict[int, float]]):
    return [
        {k: rep[1]/(v*k) for k, v in rep.items()} for rep in reps
    ]


def serial_fraction(p, psi):
    if p == 1:
        return 1.0
    else:
        return (1/psi - 1/p)/(1 - 1/p)


def transform_serial_fraction(reps: list[dict[int, float]]):
    return [
        {k: serial_fraction(k, v) for k, v in rep.items()} for rep in reps
    ]


def draw_error_plot(mapping, label):
    for k, v in mapping.items():
        x = np.array([v for v in v.keys()])
        y = np.array([v for v, _ in v.values()])
        e = np.array([v for _, v in v.values()])

        plt.errorbar(x, y, e, linestyle='None', marker='.', label=f'{k}')
        plt.legend()

    plt.xticks(np.arange(1, 13, 1.0))
    plt.xlabel('processor count')
    plt.ylabel(label)
    plt.show()


if __name__ == "__main__":
    data = split_data(load_formatted("out.txt"))
    data_speedup = {k: transform_speedup(v) for k, v in data.items()}
    data_efficiency = {k: transform_efficiency(v) for k, v in data.items()}
    data_serial_fraction = {k: transform_serial_fraction(v) for k, v in data_speedup.items()}

    data_time_for_proc = {k: x_for_proc(v) for k, v in data.items()}
    data_speedup_for_proc = {k: x_for_proc(v) for k, v in data_speedup.items()}
    data_efficiency_for_proc = {k: x_for_proc(v) for k, v in data_efficiency.items()}
    data_serial_fraction_for_proc = {k: x_for_proc(v) for k, v in data_serial_fraction.items()}

    mean_std_time_for_proc = {k: mean_std_for_proc(v) for k, v in data_time_for_proc.items()}
    mean_std_speedup_for_proc = {k: mean_std_for_proc(v) for k, v in data_speedup_for_proc.items()}
    mean_std_efficiency_for_proc = {k: mean_std_for_proc(v) for k, v in data_efficiency_for_proc.items()}
    mean_std_serial_fraction_for_proc = {k: mean_std_for_proc(v) for k, v in data_serial_fraction_for_proc.items()}

    draw_error_plot(mean_std_time_for_proc, "time [s]")
    draw_error_plot(mean_std_speedup_for_proc, "speedup")
    draw_error_plot(mean_std_efficiency_for_proc, "efficiency")
    draw_error_plot(mean_std_serial_fraction_for_proc, "serial_fraction")