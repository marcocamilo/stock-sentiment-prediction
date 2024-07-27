import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#  ────────────────────────────────────────────────────────────────────
#   IMPORT DATA
#  ────────────────────────────────────────────────────────────────────
rnn_models = pd.read_csv("./output/rnn-results.csv", index_col=0)
finbert_rnn_models = pd.read_csv("./output/finbert-rnn-results.csv", index_col=0)

#  ────────────────────────────────────────────────────────────────────
#   CONCATENATE DATA
#  ────────────────────────────────────────────────────────────────────
finbert_rnn_models.insert(2, "Sentiment", 1)
rnn_models.insert(2, "Sentiment", 0)

model_performance = pd.concat([rnn_models, finbert_rnn_models])
main_cols = ["Stock", "Model", "Sentiment", "Accuracy", "RMSE"]

print(model_performance)
#  ────────────────────────────────────────────────────────────────────
#   VISUALIZE PERFORMANCE
#  ────────────────────────────────────────────────────────────────────
g = sns.catplot(data=model_performance, x="Model", y="RMSE", hue="Stock", kind="bar", 
            col="Sentiment")
for ax in g.axes.flat:
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.2f')
plt.show()

g.savefig('./output/model-rmse-comparison.png')
