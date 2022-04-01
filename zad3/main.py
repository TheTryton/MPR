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
    df = pd.read_csv('out.csv', sep='\s*,\s*',
                     converters={'roughness': booleator, 'unstab': booleator},
                     engine='python')
    df = df.rename(columns={
        "Problem Size": "size",
        "Schedule Type": "schedule",
        "Time Taken (Mean) [s]": "mean",
        "Time Taken STD [s]": "std"
    })

    schedule_types = df['schedule'].unique()

    for schedule_type in schedule_types:
        filtered = df[df['schedule'] == schedule_type]
        plt.errorbar(
            filtered['size'], filtered['mean'],
            filtered['std'], label=schedule_type
        )

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('time taken [s]')
    plt.title('schedule')
    plt.legend()
    plt.show()

    print(df.loc[df.groupby('size')['mean'].idxmin()])

    sizes = df['size'].unique()

    n = len(sizes)
    r = np.arange(n)

    for i, size in enumerate(sizes):
        filtered = df[df['size'] == size]
        plt.bar(
            filtered['schedule'], filtered['mean'],
            yerr=filtered['std']
        )
        plt.setp(plt.xticks()[1], rotation=90)
        plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
        plt.title(f'size={size}')
        plt.yscale('log')
        plt.ylabel("time taken [s]")
        plt.show()
