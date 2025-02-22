import numpy as np
from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_features, n_estimators=100, tree_params=dict(max_depth=20, min_samples_split=10, bagging=True), bagging=True):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.tree_params = tree_params
        self.estimators = []
    
    def build_forest(self, data):
        for _ in range(self.n_estimators):
            new_tree = DecisionTree.DecisionTree(**self.tree_params)
            new_tree.build_tree(data, self.n_features)

            self.estimators.append(new_tree)

    # def predict(self, data):
    #     count

       
