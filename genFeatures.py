from itertools import combinations_with_replacement, combinations
from sklearn.base import TransformerMixin
import numpy as np

class AddFeatures(TransformerMixin):
    """
    This transformer adds combinations of features.
    """

    def fit(self, X, y=None):
        """
        This method doesn't learn any parameters.
        """
        return self

    def transform(self, X):
        """
        This method generates new features from existing ones.
        """
        columns =  X.columns
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        n_samples, n_features = X.shape
        result = []
        self.feature_names= []

        for combo in combinations_with_replacement(range(n_features), 2):
            if combo[0] == combo[1]:
                new_feature = X[:, combo[0]]
                name = f'{columns[combo[0]]}'
            else:
                new_feature = X[:, combo[0]] + X[:, combo[1]]
                name = f'{columns[combo[0]]+"+"+columns[combo[1]]}'
                
            new_feature = new_feature.reshape(-1, 1)
            result.append(new_feature)
            self.feature_names.append(name)
        
        for combo in combinations(range(n_features), 2):
            new_feature = np.abs(X[:, combo[0]] - X[:, combo[1]]) 
            new_feature = new_feature.reshape(-1, 1)
            result.append(new_feature)
            self.feature_names.append(f'{columns[combo[0]]+"-"+columns[combo[1]]}')

        for combo in combinations(range(n_features), 2):
            divisor = X[:, combo[1]]
            # Avoid division by zero
            divisor[divisor == 0] = np.nan  # or any other value you prefer
            new_feature = X[:, combo[0]] / divisor
            new_feature = new_feature.reshape(-1, 1)
            result.append(new_feature)
            self.feature_names.append(f'{columns[combo[0]]} div {columns[combo[1]]}')

        return np.hstack(result)

    def get_feature_names_out(self, input_features=None):
        """
        This method gets the names of the output features.
        """
        if not input_features:
            input_features = [str(i) for i in self.feature_names]
        feature_names = input_features.copy()

        return feature_names
