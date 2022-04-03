import matplotlib.pyplot as plt
import os

def hist_font(ax, bars=None):
    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    # Add text annotations to the top of the bars.
    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            int(round(bar.get_height(), 1)),
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
        )


def create_his(df, target, output_path):
    count_dict = dict(df[target].value_counts())
    labels, counts = [], []
    for key in count_dict.keys():
        labels.append(str(key))
        counts.append(count_dict[key])
    labels.append("None")
    counts.append(df[target].isna().sum())
    fig, ax = plt.subplots()
    plt.ylim([0, max(counts) + 0.1 * max(counts)])
    bars = ax.bar(labels, counts)
    hist_font(ax, bars)

    # Add labels and a title.
    ax.set_xlabel('Classes', labelpad=15, color='#333333')
    ax.set_ylabel("Count", labelpad=15, color='#333333')
    ax.set_title(target, pad=15, color='#333333',
                 weight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(output_path, str(target)+'.png'))

def create_his_num(df, target, output_path):
    df[target]=df[target].fillna(-10)
    fig, ax = plt.subplots()
    n,bins,patches=ax.hist(df[target],bins=15)
    plt.ylim([0,max(patches, key=lambda x: x.get_height()).get_height()*1.2])
    hist_font(ax,patches)
    ax.set_xlabel('Values', labelpad=15, color='#333333')
    ax.set_ylabel("Count", labelpad=15, color='#333333')
    ax.set_title(target, pad=15, color='#333333',
                 weight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, str(target)+'.png'))
