import matplotlib.pyplot as plt

def create_his(df,target):
    plt.hist(df[target])
    plt.show()