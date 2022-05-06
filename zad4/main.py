import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def booleator(col):
    if str(col).lower() in ['true', 'yes']:
        return True
    # elif str(col).lower() == "false":
    #    return False
    else:
        return False


if __name__ == "__main__":
    unfiltered = pd.read_csv('out.csv', sep='\s*,\s*',
                     converters={'roughness': booleator, 'unstab': booleator},
                     engine='python')

    lengths = unfiltered['length'].unique()
    dbsizes = unfiltered['desired bucket size'].unique()
    fbcoeffs = unfiltered['final bucket size coeff'].unique()
    apresets = unfiltered['allocation preset'].unique()
    version_bucket_types = unfiltered[['algorithm version', 'bucket type']].drop_duplicates()
    num_threads = unfiltered['num_threads'].unique()

    for length in lengths:
        flength = unfiltered[unfiltered['length'] == length]
        for dbsize in dbsizes:
            fdbsize = flength[flength['desired bucket size'] == dbsize]
            for fbcoeff in fbcoeffs:
                ffbcoeff = fdbsize[fdbsize['final bucket size coeff'] == fbcoeff]
                for apreset in apresets:
                    fapreset = ffbcoeff[ffbcoeff['allocation preset'] == apreset]
                    for _, vbt in version_bucket_types.iterrows():
                        fvbt = fapreset[fapreset[['algorithm version', 'bucket type']].apply(tuple, 1) == tuple(vbt)]
                        x = num_threads
                        y = list(fvbt['time (mean)'])
                        yerr = list(fvbt['time (std)'])
                        if len(x) != len(y):
                            continue
                        plt.errorbar(x, y, yerr, label=str(tuple(vbt)))
                    plt.title(f'Length={length},'
                              f' desired bucket size={dbsize},\n'
                              f' final bucket size coeff={fbcoeff},'
                              f' allocation preset={apreset}')
                    plt.ylabel('time taken [ms]')
                    plt.xlabel('thread count')
                    plt.legend()
                    plt.savefig(f'time_l_{length}_dbs_{dbsize}_fbc_{fbcoeff}_ap_{apreset}.png', dpi=100)

    exit()

    schedule_types = df['schedule'].unique()
    thread_counts = df['thread_count'].unique()

    for thread_count in thread_counts:
        filtered_tc = df[df['thread_count'] == thread_count]
        for schedule_type in schedule_types:
            filtered = filtered_tc[filtered_tc['schedule'] == schedule_type]
            plt.errorbar(
                filtered['size'], filtered['mean'],
                filtered['std'], label=schedule_type
            )
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('time taken [s]')
        plt.xlabel('problem size')
        plt.title(f'Thread Count={thread_count}')
        plt.legend()
        plt.savefig(f'thread_count_{thread_count}.png', dpi=100)
        plt.clf()

    sizes = df['size'].unique()

    for thread_count in thread_counts:
        filtered_tc = df[df['thread_count'] == thread_count]
        for size in sizes:
            filtered = filtered_tc[filtered_tc['size'] == size]
            plt.bar(
                filtered['schedule'], filtered['mean'],
                yerr=filtered['std']
            )
            plt.setp(plt.xticks()[1], rotation=90)
            plt.tight_layout(rect=[0.1, 0.03, 1, 0.95])
            plt.title(f'thread_count={thread_count} size={size}')
            plt.yscale('log')
            plt.ylabel("time taken [s]")
            plt.savefig(f'thread_count_{thread_count}_size_{size}.png', dpi=100)
            plt.clf()

    for size in sizes:
        filtered_ps = df[df['size'] == size]
        for schedule_type in schedule_types:
            filtered_st = filtered_ps[filtered_ps['schedule'] == schedule_type]
            single = filtered_st.loc[filtered_st['thread_count'].idxmin()]['mean']
            rcp_single = 1 / single
            thread_count = filtered_st['thread_count']
            speedup = single / filtered_st['mean']
            plt.plot(thread_count, speedup, marker='.')
            plt.plot([1,2,4,6,8,12,16], [1,2,4,6,8,12,16], '--')
            plt.title(f'problem_size={size} schedule_type={schedule_type}')
            plt.xlabel("thread count")
            plt.ylabel("speedup")
            plt.xscale('log')
            plt.savefig(f'problem_size_{size}_schedule_type_{schedule_type}.png', dpi=100)
            plt.clf()



