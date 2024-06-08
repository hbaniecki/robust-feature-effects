import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Algorithm:
    def __init__(
            self,
            explainer,
            constant_variables=None    # columns not to change
        ):

        self.explainer = explainer
        self._X = explainer.data.values
        self._n, self._p = self._X.shape

        self._constant_variables_id = constant_variables
        self._numerical_variables_id = np.setdiff1d(list(range(self._p)), constant_variables)

        self.result_explanation = {'original': None, 'changed': None}
        self.result_data = None

        self.iter_losses = {'iter':[], 'loss':[], 'distance_importance':[], 'distance_ranking':[], 'distance_distribution':[]}
        self.iter_data = {'iter':[], 'data':[]}


    def attack(self, target="auto", random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        self.result_explanation['original'] = self.explainer.feature_effect(self._X)
        self.result_explanation['changed'] = np.zeros_like(self.result_explanation['original'])
        
        if target == "auto":
            _target = self.result_explanation['original']
            if _target < 0.5:
                _target = 1
            else:
                _target = 0
            self.result_explanation['target'] = _target
        else:
            self.result_explanation['target'] = target


    #:# plots 
        
    def plot_data(self, i=0, constant=True, height=2, savefig=None, show=False):
        with plt.rc_context({"legend.handlelength": 0.1}):
            _colors = sns.color_palette("Set1").as_hex()[0:2][::-1]
            if i == 0:
                _df = self.result_data
            else:
                _data_changed = pd.DataFrame(self.get_best_data(i), columns=self.explainer.data.columns)
                _df = pd.concat((self.explainer.data, _data_changed))\
                        .reset_index(drop=True)\
                        .rename(index={'0': 'original', '1': 'changed'})\
                        .assign(dataset=pd.Series(['original', 'changed'])\
                                        .repeat(self._n).reset_index(drop=True))
            if not constant and self._constant_variables_id is not None:
                _df = _df.drop(_df.columns[self._constant_variables_id], axis=1)
            ax = sns.pairplot(_df, hue='dataset', height=height, palette=_colors)
            ax._legend.set_bbox_to_anchor((0.62, 0.64))
            if savefig:
                ax.savefig(savefig, bbox_inches='tight')
            if show:
                plt.show()
            else:
                return ax

    def plot_losses(self, lw=3, figsize=(9, 6), savefig=None, show=False):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            self.iter_losses['iter'], 
            self.iter_losses['loss'], 
            color='#000000', 
            lw=lw,
            label="loss"
        )
        ax.plot(
            self.iter_losses['iter'], 
            self.iter_losses['distance_distribution'], 
            color='blue', 
            lw=lw,
            label="distance_distribution"
        )
        ax.set_title('Learning curve', fontsize=20)
        ax.set_xlabel('epoch', fontsize=16)
        ax.set_ylabel('loss', fontsize=16)
        ax.legend()
        fig.tight_layout()
        if savefig:
            fig.savefig(savefig)
        if show:
            fig.show()
        else: # return the figure
            return fig

    def plot_explanation(self, figsize=(9, 6), show=False):
        temp = pd.DataFrame(self.result_explanation)
        x = np.arange(len(self.explainer.data.columns))
        width = 0.2
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, temp["original"], width, label='original', color="blue")
        ax.bar(x, temp['changed'], width, label='changed', color="red")
        ax.bar(x + width, temp['target'], width, label='target', color="black")
        ax.set_xticks(x)
        ax.legend()
        ax.set_xticklabels(self.explainer.data.columns)
        fig.tight_layout()
        if show:
            fig.show()
        else:
            return fig