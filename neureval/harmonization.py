"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience,
         IRCCS San Raffaele Scientific Institute, Milan, Italy

Site harmonization with ComBat (Johnson et al., 2007; Fortin et al., 2017, 2018).

ComBat removes additive (location) and multiplicative (scale) site effects from each
feature, while preserving the effects of the specified covariates (e.g. age, sex).
Site effects are estimated with a parametric empirical Bayes approach, which pools
information across features.

Unlike running ComBat once on the whole dataset, this implementation separates
estimation and application, so that it can be used without leakage:

- `fit` estimates the model on the training subjects only;
- `transform` applies it to any subjects (training or validation) from the sites seen
  during training, using their own covariates;
- `calibrate` adds a site that was not in the training set (external validation),
  estimating only its location and scale parameters from a calibration sample of that
  site, while the rest of the model (pooled mean, covariate effects, pooled variance)
  stays fixed. The calibration subjects should be comparable to the training
  population (e.g. healthy controls in both) and must not be used for evaluation.

The results of `fit_transform` are identical to those of the reference implementation
neuroCombat (parametric empirical Bayes), which is used in the tests.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ComBat(BaseEstimator):
    """
    ComBat harmonization with separate estimation (`fit`) and application (`transform`).

    :param eb: use empirical Bayes to estimate the site effects, default True. With False,
        the site effects are the plain per-site means and variances of the standardized data.
    :type eb: bool
    :param mean_only: correct only the site means (location), not the variances (scale),
        default False.
    :type mean_only: bool

    Attributes after fitting:

    - sites_: labels of the sites known to the model (training and calibrated sites);
    - grand_mean_: pooled mean of each feature, (n_features,);
    - beta_covariates_: effects of the preserved covariates, (n_covariates, n_features);
    - var_pooled_: pooled residual variance of each feature, (n_features,);
    - gamma_star_, delta_star_: location and scale effect of each site, (n_sites, n_features);
    - calibrated_sites_: labels of the sites added with `calibrate`.
    """

    def __init__(self, eb=True, mean_only=False):
        self.eb = eb
        self.mean_only = mean_only

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def fit(self, X, site, covariates=None):
        """
        Estimate the ComBat model on training data.

        :param X: training data, (n_subjects, n_features).
        :type X: array-like or DataFrame
        :param site: site of each subject, (n_subjects,). Any labels (strings or numbers).
        :type site: array-like
        :param covariates: numerical covariates whose effects must be preserved (e.g. age,
            sex coded 0/1), (n_subjects, n_covariates). Categorical variables with more
            than two levels must be one-hot encoded (e.g. with pandas.get_dummies,
            drop_first=True). Default None (no covariates).
        :type covariates: array-like or DataFrame
        :return: self
        """
        X = _check_data(X)
        site = _check_site(site, X.shape[0])
        C = self._check_covariates(covariates, X.shape[0], fitting=True)

        sites, site_idx, counts = np.unique(site, return_inverse=True, return_counts=True)
        if len(sites) < 2:
            raise ValueError(f"ComBat needs at least 2 sites in the training data, found {len(sites)} "
                             f"({sites.tolist()}).")
        small = sites[counts < 2]
        if len(small) > 0:
            raise ValueError(f"Each site needs at least 2 subjects in the training data; sites with "
                             f"fewer: {small.tolist()}. In cross-validation, stratify the folds by site "
                             f"(strat_vect=site), or merge or exclude very small sites.")

        n, k = X.shape[0], len(sites)
        site_onehot = np.zeros((n, k))
        site_onehot[np.arange(n), site_idx] = 1
        design = np.hstack([site_onehot, C])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("The covariates to preserve are collinear with the sites (e.g. a covariate "
                             "that is constant within each site, or present in only one site): their "
                             "effects cannot be separated from the site effects.")

        B = np.linalg.lstsq(design, X, rcond=None)[0]
        self.grand_mean_ = (counts / n) @ B[:k]
        self.beta_covariates_ = B[k:]
        var_pooled = np.mean((X - design @ B) ** 2, axis=0)
        # Features perfectly explained by site and covariates: avoid dividing by zero
        var_pooled[var_pooled == 0] = 1.0
        self.var_pooled_ = var_pooled

        s_data = self._standardize(X, C)
        self.sites_ = sites
        self.gamma_star_, self.delta_star_ = self._site_effects(s_data, site_idx, k)
        self.calibrated_sites_ = np.array([], dtype=sites.dtype)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, site, covariates=None):
        """
        Harmonize data with the fitted model. All sites must be known to the model (seen in
        training, or added with `calibrate`).

        :param X: data to harmonize, (n_subjects, n_features).
        :type X: array-like or DataFrame
        :param site: site of each subject, (n_subjects,).
        :type site: array-like
        :param covariates: covariates of these subjects, same columns as in `fit`.
        :type covariates: array-like or DataFrame
        :return: harmonized data, (n_subjects, n_features).
        :rtype: ndarray
        """
        self._check_fitted()
        X = _check_data(X, self.n_features_in_)
        site = _check_site(site, X.shape[0])
        C = self._check_covariates(covariates, X.shape[0])

        unknown = np.setdiff1d(np.unique(site), self.sites_)
        if len(unknown) > 0:
            raise ValueError(f"Site(s) {unknown.tolist()} were not in the training data. ComBat can only "
                             f"harmonize sites seen during training. For an external validation in new "
                             f"sites, calibrate them first on a separate calibration sample "
                             f"(ComBat.calibrate, or calibration_idx in evaluate_confounds). In "
                             f"cross-validation, stratify the folds by site (strat_vect=site).")

        site_idx = np.searchsorted(self.sites_, site)
        s_data = self._standardize(X, C)
        adjusted = (s_data - self.gamma_star_[site_idx]) / np.sqrt(self.delta_star_[site_idx])
        return adjusted * np.sqrt(self.var_pooled_) + self.grand_mean_ + C @ self.beta_covariates_

    def fit_transform(self, X, site, covariates=None):
        """Fit the model on the data and harmonize the same data."""
        return self.fit(X, site, covariates).transform(X, site, covariates)

    def calibrate(self, X, site, covariates=None):
        """
        Add new sites (not in the training data) to the fitted model, for external validation.

        The model estimated in training (pooled mean, covariate effects, pooled variance) is
        kept fixed; only the location and scale effects of each new site are estimated, from
        the calibration subjects given here. The calibration subjects should be comparable
        to the training population (e.g. healthy controls in both): otherwise real
        differences between the populations are removed as if they were site effects.
        They must not be used to evaluate the model.

        :param X: data of the calibration subjects, (n_subjects, n_features).
        :type X: array-like or DataFrame
        :param site: site of each calibration subject. All sites must be new.
        :type site: array-like
        :param covariates: covariates of the calibration subjects, same columns as in `fit`.
        :type covariates: array-like or DataFrame
        :return: self
        """
        self._check_fitted()
        X = _check_data(X, self.n_features_in_)
        site = _check_site(site, X.shape[0])
        C = self._check_covariates(covariates, X.shape[0])

        new_sites, site_idx, counts = np.unique(site, return_inverse=True, return_counts=True)
        known = np.intersect1d(new_sites, self.sites_)
        if len(known) > 0:
            raise ValueError(f"Site(s) {known.tolist()} are already in the model: calibrate only sites "
                             f"that were not in the training data.")
        small = new_sites[counts < 2]
        if len(small) > 0:
            raise ValueError(f"Each calibration site needs at least 2 subjects; sites with fewer: "
                             f"{small.tolist()}.")

        s_data = self._standardize(X, C)
        gamma, delta = self._site_effects(s_data, site_idx, len(new_sites))
        sites = np.concatenate([self.sites_, new_sites])
        order = np.argsort(sites, kind='stable')
        self.sites_ = sites[order]
        self.gamma_star_ = np.vstack([self.gamma_star_, gamma])[order]
        self.delta_star_ = np.vstack([self.delta_star_, delta])[order]
        self.calibrated_sites_ = np.concatenate([self.calibrated_sites_, new_sites])
        return self

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------
    def _standardize(self, X, C):
        """Remove pooled mean and covariate effects and scale by the pooled variance."""
        return (X - self.grand_mean_ - C @ self.beta_covariates_) / np.sqrt(self.var_pooled_)

    def _site_effects(self, s_data, site_idx, n_sites):
        """Location and scale effect of each site on standardized data, (n_sites, n_features)."""
        n_features = s_data.shape[1]
        gamma_hat = np.vstack([s_data[site_idx == i].mean(axis=0) for i in range(n_sites)])
        if self.mean_only:
            delta_hat = np.ones((n_sites, n_features))
        else:
            delta_hat = np.vstack([s_data[site_idx == i].var(axis=0, ddof=1) for i in range(n_sites)])
            delta_hat[delta_hat == 0] = 1.0

        if not self.eb or n_features < 2:
            # Empirical Bayes pools information across features: not possible with one feature
            return gamma_hat, delta_hat

        gamma_bar = gamma_hat.mean(axis=1)
        t2 = gamma_hat.var(axis=1, ddof=1)
        gamma_star, delta_star = np.empty_like(gamma_hat), np.empty_like(delta_hat)
        for i in range(n_sites):
            if self.mean_only:
                gamma_star[i] = _postmean(gamma_hat[i], gamma_bar[i], 1, 1, t2[i])
                delta_star[i] = 1.0
            else:
                a, b = _aprior(delta_hat[i]), _bprior(delta_hat[i])
                gamma_star[i], delta_star[i] = _it_sol(s_data[site_idx == i], gamma_hat[i], delta_hat[i],
                                                       gamma_bar[i], t2[i], a, b)
        return gamma_star, delta_star

    def _check_covariates(self, covariates, n_subjects, fitting=False):
        if covariates is None:
            C, names = np.empty((n_subjects, 0)), []
        else:
            if isinstance(covariates, pd.Series):
                covariates = covariates.to_frame()
            names = list(covariates.columns) if isinstance(covariates, pd.DataFrame) else None
            try:
                C = np.asarray(covariates, dtype=float)
            except (TypeError, ValueError):
                raise ValueError("Covariates to preserve must be numerical. Encode categorical "
                                 "variables, e.g. sex as 0/1 or with pandas.get_dummies(drop_first=True).")
            if C.ndim == 1:
                C = C.reshape(-1, 1)
            if C.shape[0] != n_subjects:
                raise ValueError(f"Covariates have {C.shape[0]} rows but there are {n_subjects} subjects.")
            if not np.all(np.isfinite(C)):
                raise ValueError("Covariates to preserve contain missing or infinite values.")
        if fitting:
            self.covariate_names_ = names
            self.n_covariates_ = C.shape[1]
        else:
            if C.shape[1] != self.n_covariates_:
                raise ValueError(f"Expected {self.n_covariates_} covariates (as in fit), got {C.shape[1]}.")
            if names is not None and self.covariate_names_ is not None and names != self.covariate_names_:
                raise ValueError(f"Covariates must have the same columns as in fit: "
                                 f"{self.covariate_names_}, got {names}.")
        return C

    def _check_fitted(self):
        if not hasattr(self, 'gamma_star_'):
            raise ValueError("ComBat is not fitted yet: call fit first.")


# ----------------------------------------------------------------------
# Use within the NeuReval pipeline
# ----------------------------------------------------------------------
def _resolve_harmonizer(harmonization, site, harmonize_covariates=None, calibration_idx=None):
    """
    Harmonization method to use: None (no harmonization), the one specified by the user,
    or ComBat with default settings when only `site` is given.
    """
    if site is None:
        if harmonization is not None:
            raise ValueError("A harmonization method was specified but site is missing: pass the "
                             "site of each subject (site=...).")
        if harmonize_covariates is not None or calibration_idx is not None:
            raise ValueError("harmonize_covariates and calibration_idx need the site of each subject "
                             "(site=...).")
        return None
    return harmonization if harmonization is not None else ComBat()


def _check_harmonization_inputs(n_subjects, modalities, site, harmonize_covariates):
    """Check that site and covariates to preserve have one row per subject."""
    if site is None:
        return
    _check_site(site, n_subjects)
    if isinstance(harmonize_covariates, dict):
        missing = [mod for mod in modalities if mod not in harmonize_covariates]
        if missing:
            raise ValueError(f"harmonize_covariates is a dictionary but has no entry for the modalities "
                             f"{missing}. Use the same keys as modalities, or a single table for all.")
        tables = harmonize_covariates.values()
    else:
        tables = [harmonize_covariates]
    for table in tables:
        if table is not None and len(table) != n_subjects:
            raise ValueError(f"harmonize_covariates has {len(table)} rows but there are {n_subjects} "
                             f"subjects.")


def _rows(obj, idx):
    """Rows of a DataFrame, Series or array by integer position."""
    if obj is None:
        return None
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.iloc[idx]
    return np.asarray(obj)[idx]


def harmonize_modalities(harmonizer, modalities, site, harmonize_covariates, tr_idx, val_idx,
                         calibration_idx=None):
    """
    Harmonize each modality, estimating the model on the training subjects only and
    applying it to the training and validation subjects. A separate model is estimated
    for each modality.

    :param harmonizer: harmonization method, e.g. ComBat(). A copy is fitted for each modality.
    :type harmonizer: ComBat
    :param modalities: dictionary with the data of each modality, (n_subjects, n_features).
    :type modalities: dictionary
    :param site: site of each subject, (n_subjects,).
    :type site: array-like
    :param harmonize_covariates: covariates whose effects ComBat must preserve: a single
        table for all modalities, or a dictionary with one table per modality (same keys as
        `modalities`), or None.
    :type harmonize_covariates: DataFrame, array-like, dictionary or None
    :param tr_idx: training subjects (integer row positions), used to estimate the model.
    :type tr_idx: array-like
    :param val_idx: validation subjects (integer row positions).
    :type val_idx: array-like
    :param calibration_idx: subjects of new sites (not in the training set) used only to
        calibrate those sites, for external validation. Default None.
    :type calibration_idx: array-like
    :return: dictionary with the harmonized data of each modality: same shape as the input,
        training and validation rows harmonized, other rows unchanged.
    :rtype: dictionary
    """
    from sklearn.base import clone
    site = np.asarray(site)
    harmonized = {}
    for mod, data in modalities.items():
        cov = harmonize_covariates[mod] if isinstance(harmonize_covariates, dict) else harmonize_covariates
        X = np.asarray(data, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        model = clone(harmonizer).fit(X[tr_idx], site[tr_idx], _rows(cov, tr_idx))
        if calibration_idx is not None:
            model.calibrate(X[calibration_idx], site[calibration_idx], _rows(cov, calibration_idx))
        H = X.copy()
        H[tr_idx] = model.transform(X[tr_idx], site[tr_idx], _rows(cov, tr_idx))
        H[val_idx] = model.transform(X[val_idx], site[val_idx], _rows(cov, val_idx))
        harmonized[mod] = H
    return harmonized


# ----------------------------------------------------------------------
# Helper functions (empirical Bayes, as in neuroCombat)
# ----------------------------------------------------------------------
def _check_data(X, n_features=None):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"Data must be 2-D (subjects x features), got {X.ndim} dimensions.")
    if not np.all(np.isfinite(X)):
        raise ValueError("Data to harmonize contain missing or infinite values.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features (as in fit), got {X.shape[1]}.")
    return X


def _check_site(site, n_subjects):
    site = np.asarray(site)
    if site.ndim != 1 or len(site) != n_subjects:
        raise ValueError(f"site must have one value per subject ({n_subjects}), got shape {site.shape}.")
    if pd.isna(site).any():
        raise ValueError("site contains missing values.")
    return site


def _aprior(delta_hat):
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (2 * s2 + m ** 2) / s2


def _bprior(delta_hat):
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (m * s2 + m ** 3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(s_data, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001, max_iter=1000):
    """Iterative solution of the empirical Bayes estimates of one site."""
    n = s_data.shape[0]
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((s_data - g_new) ** 2).sum(axis=0)
        d_new = _postvar(sum2, n, a, b)
        change = max(np.max(np.abs(g_new - g_old) / g_old), np.max(np.abs(d_new - d_old) / d_old))
        g_old, d_old = g_new, d_new
        if change <= conv:
            break
    return g_new, d_new
