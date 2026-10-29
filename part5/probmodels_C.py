"""Probabilistic linear models for Part 2, model C (written from scratch with numpy / scipy).

1. LaplaceLinear              y = w.x + b + Laplace noise   -> maximum likelihood = least ABSOLUTE deviations
                              (median regression). Fitted by iteratively reweighted least squares (IRLS).
2. HeteroscedasticGaussian    y ~ Normal(mu(x), sigma^2(x)) with  mu = w.x + b  and  log sigma^2 = v.z + c.
                              Fitted by maximum likelihood with L-BFGS and exact gradients.
3. HeteroscedasticLaplace     y ~ Laplace(mu(x), scale(x)),  mu = w.x + b,  log scale = v.z + c  (extension).

All models standardise inputs with statistics from the TRAINING rows only.
"""
import numpy as np
from scipy.optimize import minimize


class Standardiser:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


def _with_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


class LaplaceLinear:
    """Least-absolute-deviation regression = MLE under Laplace noise, solved by IRLS.

    Laplace negative log-likelihood:  sum_i |y_i - w.x_i| / b  + n log(2b).
    Minimising over w does not involve b, so b is estimated afterwards as mean |residual|.
    IRLS: |r| = r^2 / |r|, so each step is a weighted least squares with weights 1 / max(|r|, eps).
    """

    def __init__(self, l2=1e-3, max_iter=200, tol=1e-6, eps=1e-2):
        self.l2, self.max_iter, self.tol, self.eps = l2, max_iter, tol, eps

    def fit(self, X, y, sample_weight=None):
        self.scaler_ = Standardiser().fit(X)
        design = _with_intercept(self.scaler_.transform(X))
        base_weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        penalty = self.l2 * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ (design * base_weight[:, None]) + penalty,
                                       design.T @ (base_weight * y))
        self.loss_history_ = []
        for _ in range(self.max_iter):
            residual = y - design @ coefficients
            self.loss_history_.append(np.mean(base_weight * np.abs(residual)))
            irls_weight = base_weight / np.maximum(np.abs(residual), self.eps)
            new_coefficients = np.linalg.solve(design.T @ (design * irls_weight[:, None]) + penalty,
                                               design.T @ (irls_weight * y))
            coefficients = new_coefficients
            if len(self.loss_history_) > 1 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol * self.loss_history_[-1]:
                break
        self.coef_ = coefficients
        self.scale_b_ = np.mean(np.abs(y - design @ coefficients))
        return self

    def predict(self, X):
        return _with_intercept(self.scaler_.transform(X)) @ self.coef_


class HeteroscedasticGaussian:
    """Gaussian linear model whose variance also depends on the inputs.

    NLL = 0.5 * sum [ log sigma^2_i + (y_i - mu_i)^2 / sigma^2_i ]
    d/dw = - sum (y - mu) / sigma^2 * x          d/dv = 0.5 * sum [ 1 - (y - mu)^2 / sigma^2 ] * z
    Points with large predicted variance get small weight in the mean fit (a learned weighted least squares).
    """

    def __init__(self, l2=1e-3, variance_columns=None, max_iter=500):
        self.l2, self.variance_columns, self.max_iter = l2, variance_columns, max_iter

    def _split(self, parameters, n_mean, n_var):
        return parameters[:n_mean], parameters[n_mean:n_mean + n_var]

    def fit(self, X, y, sample_weight=None):
        self.scaler_ = Standardiser().fit(X)
        scaled = self.scaler_.transform(X)
        mean_design = _with_intercept(scaled)
        columns = slice(None) if self.variance_columns is None else list(self.variance_columns)
        variance_design = _with_intercept(scaled[:, columns])
        n_mean, n_var = mean_design.shape[1], variance_design.shape[1]
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        total_weight = weight.sum()

        start_mean = np.linalg.lstsq(mean_design, y, rcond=None)[0]
        start_variance = np.zeros(n_var)
        start_variance[0] = np.log(np.mean((y - mean_design @ start_mean) ** 2))
        start = np.concatenate([start_mean, start_variance])

        def objective(parameters):
            w, v = self._split(parameters, n_mean, n_var)
            mu = mean_design @ w
            log_var = np.clip(variance_design @ v, -20, 20)
            inv_var = np.exp(-log_var)
            residual = y - mu
            nll = 0.5 * np.sum(weight * (log_var + residual ** 2 * inv_var)) / total_weight
            grad_w = -(mean_design.T @ (weight * residual * inv_var)) / total_weight
            grad_v = 0.5 * (variance_design.T @ (weight * (1 - residual ** 2 * inv_var))) / total_weight
            nll += 0.5 * self.l2 * (np.sum(w[1:] ** 2) + np.sum(v[1:] ** 2)) / len(y) * 1.0
            grad_w[1:] += self.l2 * w[1:] / len(y)
            grad_v[1:] += self.l2 * v[1:] / len(y)
            return nll, np.concatenate([grad_w, grad_v])

        result = minimize(objective, start, jac=True, method="L-BFGS-B", options=dict(maxiter=self.max_iter, maxfun=self.max_iter * 2))
        self.result_ = result
        self.mean_coef_, self.var_coef_ = self._split(result.x, n_mean, n_var)
        self._columns, self._n_mean = columns, n_mean
        return self

    def predict_mean_and_variance(self, X):
        scaled = self.scaler_.transform(X)
        mu = _with_intercept(scaled) @ self.mean_coef_
        log_var = np.clip(_with_intercept(scaled[:, self._columns]) @ self.var_coef_, -20, 20)
        return mu, np.exp(log_var)

    def predict(self, X):
        return self.predict_mean_and_variance(X)[0]


MAX_PM25 = 500.0


class LogLaplaceForecaster:
    """Laplace linear model on log1p(PM2.5), returning forecasts in ug/m3.

    mode="median": expm1(mu). The median commutes with log, so this is the exact conditional median
                   and therefore the MAE-optimal forecast.
    mode="mean":   exp(mu) * smearing - 1, where smearing = mean(exp(train residual)) (Duan, 1983).
                   This estimates the conditional mean and is the RMSE-optimal forecast.
    """

    def __init__(self, l2=1e-3, mode="mean"):
        self.l2, self.mode = l2, mode

    def fit(self, X, y_micrograms):
        log_target = np.log1p(y_micrograms)
        self.model_ = LaplaceLinear(l2=self.l2).fit(X, log_target)
        self.smearing_ = np.mean(np.exp(log_target - self.model_.predict(X)))
        return self

    def predict(self, X, mode=None):
        mu = self.model_.predict(X)
        if (mode or self.mode) == "median":
            forecast = np.expm1(mu)
        else:
            forecast = np.exp(mu) * self.smearing_ - 1.0
        return np.clip(forecast, 0.0, MAX_PM25)


class LogHeteroGaussianForecaster:
    """Heteroscedastic Gaussian model on log1p(PM2.5): log y ~ Normal(mu(x), sigma^2(x)).

    The mean forecast in ug/m3 is the log-normal mean  exp(mu + sigma^2 / 2) - 1  (uses the learned variance).
    """

    def __init__(self, l2=1e-3, variance_columns=None):
        self.l2, self.variance_columns = l2, variance_columns

    def fit(self, X, y_micrograms):
        self.model_ = HeteroscedasticGaussian(l2=self.l2, variance_columns=self.variance_columns).fit(X, np.log1p(y_micrograms))
        return self

    def predict(self, X, mode="mean"):
        mu, variance = self.model_.predict_mean_and_variance(X)
        forecast = np.expm1(mu) if mode == "median" else np.expm1(mu + variance / 2.0)
        return np.clip(forecast, 0.0, MAX_PM25)

    def predict_log_distribution(self, X):
        mu, variance = self.model_.predict_mean_and_variance(X)
        return mu, np.sqrt(variance)
