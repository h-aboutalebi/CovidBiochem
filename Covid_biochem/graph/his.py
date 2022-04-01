import matplotlib.pyplot as plt


def create_his(df, target):
    count_dict=dict(df[target].value_counts())
    labels, counts=[],[]
    for key in count_dict.keys():
        labels.append(key)
        counts.append(count_dict[key])
    fig, ax = plt.subplots()
    plt.ylim([0, max(counts)+0.2*max(counts)])
    labels.append("None")
    plt.title(target)
    plt.ylabel("count")
    counts.append(df[target].isna().sum())
    bars = ax.bar(labels, counts)
    # Axis formatting.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Grab the color of the bars so we can make the
    # text the same color.
    bar_color = bars[0].get_facecolor()

    # Add text annotations to the top of the bars.
    # Note, you'll have to adjust this slightly (the 0.3)
    # with different data.
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
        )

    fig.tight_layout()
    plt.show()

