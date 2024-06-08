import numpy as np
import pandas as pd
import torch

class Explainer:
    def __init__(
        self,
        model,
        xs,
        X,
        s,
        method
    ):
        if method not in ("pd", "cd", "dale"):
            raise ValueError("`method` should be one of {'pd', 'cd', 'dale'}")

        self.model = model
        self.s = s
        self.xs = xs
        self.method = method
        self.data = pd.DataFrame(X)
        if self.method == "dale":
            from captum.attr import Saliency
            self.explainer = Saliency(model)

    #########
        
    def feature_effect(self, X):
        if self.method == "pd":
            return self.partial_dependence(X)
        elif self.method == "cd":
            return self.conditional_dependence(X)
        elif self.method == "dale":
            return self.dale(X)
    
    def feature_effect_pop(self, X):
        if self.method == "pd":
            return self.partial_dependence_pop(X)
        elif self.method == "cd":
            return self.conditional_dependence_pop(X)
        elif self.method == "dale":
            return self.dale_pop(X)
        
    #########
        
    def partial_dependence(self, X):
        X_raw = X.copy()
        X_raw[:, self.s] = self.xs
        return self.model.predict_proba(X_raw)[:, 1].mean()

    def conditional_dependence(self, X):
        X_raw = X.copy()
        X_s = X_raw[:, self.s]
        epsilon = X_s.ptp() / 18
        X_cond = X_raw[(X_s > self.xs - epsilon) & (X_s < self.xs + epsilon), :]
        if X_cond.shape[0] == 0:
            return self.partial_dependence(X)
        else:
            X_cond[:, self.s] = self.xs
            return self.model.predict_proba(X_cond)[:, 1].mean()
        
    def dale(self, X):
        X_raw = X.copy()
        X_s = X_raw[:, self.s]
        x_smin = X_s.min()
        delta_x = X_s.ptp() / 18
        k = 0
        mu_k = 0
        while x_smin + k * delta_x < self.xs:
            S_k = (X_s > x_smin + k * delta_x) & (X_s < x_smin + (k+1) * delta_x)
            if np.any(S_k):
                X_k = X[S_k, :]
                inputs = torch.as_tensor(X_k, dtype=torch.float)
                inputs.requires_grad_()
                mu_k += self.explainer.attribute(inputs, target=1, abs=False)[:, self.s].mean().item()
            k += 1
        return delta_x * mu_k        

    #########
        
    def partial_dependence_pop(self, X):
        X_long = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
        return np.apply_along_axis(lambda x, d: self.partial_dependence(x.reshape((d[0], d[1]))), 1, X_long, d=X[0].shape)

    def conditional_dependence_pop(self, X):
        X_long = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
        return np.apply_along_axis(lambda x, d: self.conditional_dependence(x.reshape((d[0], d[1]))), 1, X_long, d=X[0].shape)

    def dale_pop(self, X):
        X_long = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
        return np.apply_along_axis(lambda x, d: self.dale(x.reshape((d[0], d[1]))), 1, X_long, d=X[0].shape)