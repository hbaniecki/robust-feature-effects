import numpy as np
import pandas as pd
import tqdm

from . import algorithm     
from . import loss
from . import utils


class GeneticAlgorithm(algorithm.Algorithm):
    def __init__(
        self,
        explainer,
        constant_variables=None,
        **kwargs
    ):
        super().__init__(
            explainer=explainer,
            constant_variables=constant_variables
        )
        
        params = dict(
            epsilon=1e-4,
            stop_iter=20,
            pop_count=50,
            std_ratio=1/9,
            mutation_prob=0.5,
            mutation_with_constraints=True,
            crossover_ratio=0.5,
            top_survivors=2
        )
        
        for k, v in kwargs.items():
            params[k] = v

        self.params = params
        
        # prepare std vector for mutation
        if self._numerical_variables_id is not None:
            self._numerical_std = np.nanstd(self._X[:, self._numerical_variables_id], axis=0) * params['std_ratio']
            self._numerical_ptp = {v: np.nanmax(self._X[:, v]) - np.nanmin(self._X[:, v]) for v in self._numerical_variables_id}
            # self._numerical_bin = {v: np.histogram_bin_edges(self._X[:, v], bins='auto') for v in self._numerical_variables_id}
            if params['mutation_with_constraints']:
                self._numerical_minmax = {v: {
                    'min': np.nanmin(self._X[:, v]), 
                    'max': np.nanmax(self._X[:, v]),
                    'std': np.nanstd(self._X[:, v])
                } for v in self._numerical_variables_id}
        
        # calculate probs for rank selection method
        self._rank_probs = np.arange(params['pop_count'], 0, -1) /\
             (params['pop_count'] * (params['pop_count'] + 1) / 2)
        

    #:# algorithm
  
    def attack(
        self,
        target="auto",
        random_state=None,
        max_iter=50,
        cache_data=False,
        verbose=True
    ):
        
        super().attack(target=target, random_state=random_state)
            
        # init population
        self._X_pop = np.tile(self._X, (self.params['pop_count'], 1, 1))
        self._E_pop = np.tile(self.result_explanation['original'], self.params['pop_count']) 
        self._L_pop = np.zeros(self.params['pop_count']) 
        self.mutation(adjust=3)
        self.append_losses(iter=0)
        _curr_best = self.iter_losses['loss'][-1]
        
        if cache_data:
            self.append_data(iter=0)

        pbar = tqdm.tqdm(range(1, max_iter + 1), disable=not verbose)
        for iter in pbar:
            self.crossover()
            self.mutation()
            self.evaluation()
            if iter != max_iter:
                self.selection()

            self.append_losses(iter=iter)
            if cache_data:
                if self.iter_losses['loss'][-1] < _curr_best:
                    _curr_best = self.iter_losses['loss'][-1]
                    self.append_data(iter=iter)
            pbar.set_description("Iter: %s || Loss: %s" % (iter, self.iter_losses['loss'][-1]))
            if utils.check_early_stopping(self.iter_losses, self.params['epsilon'], self.params['stop_iter']):
                if verbose:
                    print("Breaking due to early stopping in Iter:", iter, "|| Loss:", _curr_best if cache_data else self.iter_losses['loss'][-1])
                break
        
        if cache_data:
            iter, _X_changed, = self.get_best_cached()
            if verbose:
                print("Retrieving best cached dataset from Iter:", iter, "|| Loss:", _curr_best)
        else:
            _X_changed = self.get_best_data()
    
        self.result_explanation['changed'] = self.explainer.feature_effect(_X_changed)

        _data_changed = pd.DataFrame(_X_changed, columns=self.explainer.data.columns)
        
        self.result_data = pd.concat((self.explainer.data, _data_changed))\
            .reset_index(drop=True)\
            .rename(index={'0': 'original', '1': 'changed'})\
            .assign(dataset=pd.Series(['original', 'changed'])\
                            .repeat(self._n).reset_index(drop=True))
            

    #:# inside
    
    def mutation(self, adjust=1):   
        #:# change numerical variables with gaussian noise

        _temp_pop_count = self._X_pop.shape[0]  
        # individual mask made with the probability 
        _idpop = np.flatnonzero(np.random.binomial(
            n=1,
            p=self.params['mutation_prob'], 
            size=_temp_pop_count
        ))

        if self._numerical_variables_id is not None:       
            _temp_var_count = len(self._numerical_variables_id)
            _theta = np.random.normal(
                loc=0,
                scale=self._numerical_std * adjust,
                size=(len(_idpop), self._n, _temp_var_count)
            )
            # column mask made with the probability 
            _mask = np.random.binomial(
                n=1,
                p=self.params['mutation_prob'], 
                size=(len(_idpop), 1, _temp_var_count)
            )
            self._X_pop[np.ix_(_idpop, np.arange(self._n), self._numerical_variables_id)] += _theta * _mask
            
            if self.params['mutation_with_constraints']:
                # add min/max constraints for the variable distribution
                _X_pop_long = self._X_pop.reshape(_temp_pop_count * self._n, self._p)
                for v in self._numerical_variables_id:
                    _max_mask = np.flatnonzero(_X_pop_long[:, v] > self._numerical_minmax[v]['max'])
                    _min_mask = np.flatnonzero(_X_pop_long[:, v] < self._numerical_minmax[v]['min'])
                    if len(_max_mask) > 0:
                        _X_pop_long[:, v][_max_mask] = np.random.uniform(
                            self._numerical_minmax[v]['max'] - self._numerical_minmax[v]['std'],
                            self._numerical_minmax[v]['max'],
                            size=len(_max_mask)
                        )
                    if len(_min_mask) > 0:
                        _X_pop_long[:, v][_min_mask] = np.random.uniform(
                            self._numerical_minmax[v]['min'],
                            self._numerical_minmax[v]['min'] + self._numerical_minmax[v]['std'],
                            size=len(_min_mask)
                        )

                self._X_pop = _X_pop_long.reshape(_temp_pop_count, self._n, self._p)
    

    def crossover(self):
        #:# exchange values between the individuals

        # indexes of the population subset
        _idpop = np.random.choice(
            self.params['pop_count'], 
            size=int(self.params['pop_count'] * self.params['crossover_ratio']),
            replace=False
        )
        # get parents
        _parents = self._X_pop[_idpop, :, :].copy()
        # indexes of values
        _size = np.prod(_parents.shape)
        _idval = np.random.choice(_size, int(_size/2), replace=False)
        # create childs
        _childs = _parents.copy().reshape(_size)
        _childs[_idval] = _parents[::-1, :, :].reshape(_size)[_idval]
        _childs = _childs.reshape(_parents.shape)

        self._X_pop = np.concatenate((self._X_pop, _childs))
    

    def evaluation(self):
        #:# calculate explanations and distances

        _t = self.result_explanation['target']
        self._E_pop = self.explainer.feature_effect_pop(self._X_pop)
        _loss = loss.loss_pop(arr1=_t, arr2=self._E_pop)

        _L = _loss
        self._L_pop = _L
            

    def selection(self):
        #:# take n best individuals and use p = i/(n*(n-1))

        _top_survivors = self.params['top_survivors']
        _top_f_ids = np.argpartition(self._L_pop, _top_survivors)[:_top_survivors]
        # _top_f_ids = np.repeat(np.argmin(self._L_pop), _top_survivors)
        _random_ids = np.random.choice(
            self.params['pop_count'], 
            size=self.params['pop_count'] - _top_survivors, 
            replace=True,
            p=self._rank_probs
        )
        _sorted_ids = np.argsort(self._L_pop)[_random_ids]
        self._X_pop = np.concatenate((
            self._X_pop[_sorted_ids],
            self._X_pop[_top_f_ids]
        ))
        self._L_pop = np.concatenate((
            self._L_pop[_sorted_ids],
            self._L_pop[_top_f_ids]
        ))
        assert self._X_pop.shape[0] == self.params['pop_count'], 'wrong selection'


    #:# helper
    
    def get_best_id(self, i=0):
        return np.argsort(self._L_pop)[i]
        
    def get_best_data(self, i=0):
        return self._X_pop[np.argsort(self._L_pop)][i]

    def get_best_cached(self):
        return self.iter_data['iter'][-1], self.iter_data['data'][-1]

    def append_data(self, iter=0):
        self.iter_data['iter'].append(iter)
        self.iter_data['data'].append(self.get_best_data())

    def append_losses(self, iter=0):
        _best = self.get_best_id()
        _t = self.result_explanation['target']
        _loss = loss.loss(
            arr1=_t,
            arr2=self._E_pop[_best]
        )
        _L = _loss
        self.iter_losses['iter'].append(iter)
        self.iter_losses['loss'].append(_L)