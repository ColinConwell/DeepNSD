#modified from: github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py

import numpy as np
from scipy import sparse

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.base import RegressorMixin, MultiOutputMixin, is_classifier
from sklearn.linear_model._ridge import safe_sparse_dot, _RidgeGCV, _BaseRidgeCV
from sklearn.linear_model._ridge import _rescale_data, _preprocess_data
from sklearn.linear_model._ridge import _check_gcv_mode, _check_sample_weight

from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score

pearsonr_vec = np.vectorize(pearsonr, signature='(n),(n)->(),()')

def pearson_r_score(y_true, y_pred, multioutput=None):
    y_true_ = y_true.transpose()
    y_pred_ = y_pred.transpose()
    return(pearsonr_vec(y_true_, y_pred_)[0])

class _RidgeGCVMod(_RidgeGCV):
    """Ridge regression with built-in Leave-one-out Cross-Validation."""
    
    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        #normalize="deprecated",
        scoring=None,
        copy_X=True,
        gcv_mode=None,
        store_cv_values=False,
        is_clf=False,
        alpha_per_target=False,
    ):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        #self.normalize = normalize
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
        self.is_clf = is_clf
        self.alpha_per_target = alpha_per_target

    def fit(self, X, y, sample_weight=None):
        _normalize = False

        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            dtype=[np.float64],
            multi_output=True,
            y_numeric=True,
        )

        # alpha_per_target cannot be used in classifier mode. All subclasses
        # of _RidgeGCV that are classifiers keep alpha_per_target at its
        # default value: False, so the condition below should never happen.
        assert not (self.is_clf and self.alpha_per_target)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        if np.any(self.alphas <= 0):
            raise ValueError(
                "alphas must be strictly positive. Got {} containing some "
                "negative or null value instead.".format(self.alphas)
            )

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            self.fit_intercept,
            #_normalize,
            self.copy_X,
            sample_weight=sample_weight,
        )

        gcv_mode = _check_gcv_mode(X, self.gcv_mode)

        if gcv_mode == "eigen":
            decompose = self._eigen_decompose_gram
            solve = self._solve_eigen_gram
        elif gcv_mode == "svd":
            if sparse.issparse(X):
                decompose = self._eigen_decompose_covariance
                solve = self._solve_eigen_covariance
            else:
                decompose = self._svd_decompose_design_matrix
                solve = self._solve_svd_design_matrix

        n_samples = X.shape[0]

        if sample_weight is not None:
            X, y = _rescale_data(X, y, sample_weight)
            sqrt_sw = np.sqrt(sample_weight)
        else:
            sqrt_sw = np.ones(n_samples, dtype=X.dtype)

        X_mean, *decomposition = decompose(X, y, sqrt_sw)

        if self.scoring not in ['pearson_r', 'explained_variance']:
            raise ValueError("modified RidgeCV scoring requires one of ['pearson_r','explained_variance']")

        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        n_alphas = 1 if np.ndim(self.alphas) == 0 else len(self.alphas)

        if self.store_cv_values:
            self.cv_values_ = np.empty((n_samples * n_y, n_alphas), dtype=X.dtype)

        best_coef, best_score, best_alpha = None, None, None

        for i, alpha in enumerate(np.atleast_1d(self.alphas)):
            G_inverse_diag, c = solve(float(alpha), y, sqrt_sw, X_mean, *decomposition)
            predictions = y - (c / G_inverse_diag)
            if self.store_cv_values:
                self.cv_values_[:, i] = predictions.ravel()

            #identity_estimator = _IdentityRegressor()
            if self.alpha_per_target:
                if self.scoring == 'pearson_r':
                    alpha_score = pearson_r_score(y, predictions)
                if self.scoring == 'explained_variance':
                    alpha_score = explained_variance_score(y, predictions, multioutput = 'raw_values') 
            else:
                if self.scoring == 'pearson_r':
                    alpha_score = pearson_r_score(y, predictions).mean()
                if self.scoring == 'explained_variance':
                    alpha_score = explained_variance_score(y, predictions, multioutput = 'uniform_average')

            # Keep track of the best model
            if best_score is None: 
                if self.alpha_per_target and n_y > 1:
                    best_coef = c
                    best_score = np.atleast_1d(alpha_score)
                    best_alpha = np.full(n_y, alpha)
                else:
                    best_coef = c
                    best_score = alpha_score
                    best_alpha = alpha
            else: 
                if self.alpha_per_target and n_y > 1:
                    to_update = alpha_score > best_score
                    best_coef[:, to_update] = c[:, to_update]
                    best_score[to_update] = alpha_score[to_update]
                    best_alpha[to_update] = alpha
                elif alpha_score > best_score:
                    best_coef, best_score, best_alpha = c, alpha_score, alpha

        self.alpha_ = best_alpha
        self.best_score_ = best_score
        self.dual_coef_ = best_coef
        self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)

        X_offset += X_mean * X_scale
        self._set_intercept(X_offset, y_offset, X_scale)

        if self.store_cv_values:
            if len(y.shape) == 1:
                cv_values_shape = n_samples, n_alphas
            else:
                cv_values_shape = n_samples, n_y, n_alphas
            self.cv_values_ = self.cv_values_.reshape(cv_values_shape)

        return self


class _BaseRidgeCVMod(_BaseRidgeCV):
    def fit(self, X, y, sample_weight=None):
        cv = self.cv
        if cv is None:
            estimator = _RidgeGCVMod(
                self.alphas,
                fit_intercept=self.fit_intercept,
                #normalize=self.normalize,
                scoring=self.scoring,
                gcv_mode=self.gcv_mode,
                store_cv_values=self.store_cv_values,
                is_clf=is_classifier(self),
                alpha_per_target=self.alpha_per_target,
            )
            estimator.fit(X, y, sample_weight=sample_weight)
            self.alpha_ = estimator.alpha_
            self.best_score_ = estimator.best_score_
            if self.store_cv_values:
                self.cv_values_ = estimator.cv_values_
        else:
            if self.store_cv_values:
                raise ValueError("cv is not None and store_cv_values=True are incompatible")
            if self.alpha_per_target:
                raise ValueError("cv is not None and alpha_per_target=True are incompatible")
            
            parameters = {"alpha": self.alphas}
            solver = "sparse_cg" if sparse.issparse(X) else "auto"
            model = RidgeClassifier if is_classifier(self) else Ridge
            gs = GridSearchCV(
                model(
                    fit_intercept=self.fit_intercept,
                    #normalize=self.normalize,
                    solver=solver,
                ),
                parameters,
                cv=cv,
                scoring=self.scoring,
            )
            gs.fit(X, y, sample_weight=sample_weight)
            estimator = gs.best_estimator_
            self.alpha_ = gs.best_estimator_.alpha
            self.best_score_ = gs.best_score_

        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.n_features_in_ = estimator.n_features_in_
        if hasattr(estimator, "feature_names_in_"):
            self.feature_names_in_ = estimator.feature_names_in_

        return self


class RidgeCVMod(MultiOutputMixin, RegressorMixin, _BaseRidgeCVMod):
    """Ridge regression with built-in cross-validation."""
    
if __name__ == "__main__":
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 1)
    
    regression = RidgeCVMod(alphas = np.logspace(-3, 3, 100), scoring = 'pearson_r')
    regression.fit(x, y)

    print('Best score:', regression.best_score_)
    print('Best alpha:', regression.alpha_)
