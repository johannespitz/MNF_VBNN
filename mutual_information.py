import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def _compute_mi(model, samples, num_runs):
    predictions_list = []
    for idx in tqdm(range(num_runs), file=sys.stdout):
        predictions_list.append(model.predict(samples, batch_size=10000))

    probs = np.asarray(predictions_list)
    mean = probs.mean(axis=0)

    def _entropy(ps):
        return -1. * (ps * np.log(ps + 1e-8)).sum(axis=-1)

    mean_H = _entropy(mean)
    indi_H = _entropy(probs)

    return mean_H - indi_H.mean(axis=0), mean_H

def plot_mi(tests_dict, model, num_runs=20):
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    for name, samples in tests_dict.items():
        mi, ent = _compute_mi(model, samples, num_runs)
        print(f'MI: {name:15s} min: {mi.min():.4f}, max: {mi.max():.4f}, mean: {mi.mean():.4f}')
        print(f'PE: {name:15s} min: {ent.min():.4f}, max: {ent.max():.4f}, mean: {ent.mean():.4f}')
        # with self.summary_writer.as_default():
        #     tf.summary.histogram('mi/' + name, mi, epoch)
        #     tf.summary.histogram('ent/' + name, ent, epoch)
        sns.distplot(mi, label=name, ax=axs[0], kde=True, kde_kws=dict(gridsize=1000))
        sns.distplot(ent, label=name, ax=axs[1], kde=True, kde_kws=dict(gridsize=1000))

    plt.legend()
    fig.tight_layout()
    return plt