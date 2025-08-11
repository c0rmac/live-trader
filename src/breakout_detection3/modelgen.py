import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import math

class KSplit:

    def __init__(self, accuracies_val, accuracies_y1_val, accuracies_y0_val, models):
        self.accuracies_val = accuracies_val
        self.accuracies_y1_val = accuracies_y1_val
        self.accuracies_y0_val = accuracies_y0_val
        self.models = models

class KFoldResult:

    def __init__(self, kSplits: list[KSplit], val_variation: float):
        self.kSplits = kSplits
        self.val_variation = val_variation

class PredictionModel:

    def __init__(self, model_features, min_var, max_var, k_splits):
        self.model_features = model_features
        self.min_var = min_var
        self.max_var = max_var
        self.k_splits = k_splits

    def predict_from_val_threshold(self, df, threshold = 0.4, upper_y0_threshold = 0.85) -> np.array:
        index_set = [
            i for i, kSplit in enumerate(self.k_splits) if kSplit.accuracies_y0_val >= upper_y0_threshold
            and kSplit.accuracies_y1_val is not None
            and kSplit.accuracies_y1_val <= threshold
            # and accuracies_y1_test[i] != 0
        ]

        # Start with a mask of all False
        union_mask = np.zeros(len(df), dtype=bool)

        for i in index_set:
            model = self.k_splits[i].models
            preds = model.predict(df[self.model_features]).astype(np.bool)
            union_mask |= preds  # Bitwise OR for boolean union
            #union_mask &= preds  # Bitwise OR for boolean union

        return union_mask