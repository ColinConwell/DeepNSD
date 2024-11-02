import numpy as np
from typing import List, Tuple
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# Implementation of https://www.biorxiv.org/content/10.1101/2021.03.21.436284v1.full.pdf
# that computes the statistics of manifold geometry relevant for few-shot classification
# modified from: https://www.biorxiv.org/content/10.1101/2022.07.13.499969v1
# code link: https://github.com/EricElmoznino/encoder_dimensionality (lib/manifold_geometry.py)

class ManifoldGeometry:
    def __init__(self, X: np.ndarray):
        pca = PCA()
        pca.fit(X)
        squared_radii = pca.explained_variance_
        eigvecs = pca.components_

        self.num_examples = X.shape[0]
        self.centroid = X.mean(axis=0)
        self.radius = np.sqrt(squared_radii.mean())
        self.dimensionality = (squared_radii).sum() ** 2 / (squared_radii ** 2).sum()
        self.directions_of_variance = np.sqrt(squared_radii)[:, np.newaxis] * eigvecs / self.radius


def get_manifold_statistics(concept_manifolds):
    radius = manifold_radius(concept_manifolds)
    signal = manifold_signal(concept_manifolds)
    bias = manifold_bias(concept_manifolds)
    dim = manifold_dimensionality(concept_manifolds)
    sno_a, sno_b = manifold_signal_noise_overlap(concept_manifolds)
    snr = manifold_signal_noise_ratio(signal, bias, dim, sno_a, sno_b, concept_manifolds[0].num_examples)
    radius_mean, radius_std = statistics(radius)
    signal_mean, signal_std = statistics(signal)
    bias_mean, bias_std = statistics(bias)
    dim_mean, dim_std = statistics(dim)
    sno_a_mean, sno_a_std = statistics(sno_a)
    sno_b_mean, sno_b_std = statistics(sno_b)
    snr_mean, snr_std = statistics(snr)
    global_radius, global_dim = manifold_global_statistics(concept_manifolds)
    manifold_statistics = {
        'within-concept radius (mean)': radius_mean, 'within-concept radius (std)': radius_std,
        'within-concept dimensionality (mean)': dim_mean, 'within-concept dimensionality (std)': dim_std,
        'between-concept radius': global_radius, 'between-concept dimensionality': global_dim,
        'signal (mean)': signal_mean, 'signal (std)': signal_std,
        'bias (mean)': bias_mean, 'bias (std)': bias_std,
        'self signal-noise-overlap (mean)': sno_a_mean, 'self signal-noise-overlap (std)': sno_a_std,
        'other signal-noise-overlap (mean)': sno_b_mean, 'other signal-noise-overlap (std)': sno_b_std,
        'signal-noise-ratio (mean)': snr_mean, 'signal-noise-ratio (std)': snr_std
    }
    return manifold_statistics


def manifold_signal(manifolds: List[ManifoldGeometry]) -> np.ndarray:
    assert len(manifolds) > 1
    signal = np.full((len(manifolds), len(manifolds)), np.nan, dtype=manifolds[0].centroid.dtype)
    for i in range(len(manifolds) - 1):
        for j in range(i + 1, len(manifolds)):
            man_i, man_j = manifolds[i], manifolds[j]
            distance = euclidean(man_i.centroid, man_j.centroid)
            signal[i, j] = distance / man_i.radius
            signal[j, i] = distance / man_j.radius
    return signal


def manifold_bias(manifolds: List[ManifoldGeometry]) -> np.ndarray:
    assert len(manifolds) > 1
    bias = np.full((len(manifolds), len(manifolds)), np.nan, dtype=manifolds[0].centroid.dtype)
    for i in range(len(manifolds) - 1):
        for j in range(i + 1, len(manifolds)):
            man_i, man_j = manifolds[i], manifolds[j]
            bias[i, j] = man_j.radius / man_i.radius - 1
            bias[j, i] = man_i.radius / man_j.radius - 1
    return bias


def manifold_dimensionality(manifolds: List[ManifoldGeometry]) -> np.ndarray:
    assert len(manifolds) > 1
    dimensionality = np.array([m.dimensionality for m in manifolds])
    return dimensionality


def manifold_radius(manifolds: List[ManifoldGeometry]) -> np.ndarray:
    assert len(manifolds) > 1
    radius = np.array([m.radius for m in manifolds])
    return radius


def manifold_signal_noise_overlap(manifolds: List[ManifoldGeometry]) -> Tuple[np.ndarray, np.ndarray]:
    assert len(manifolds) > 1
    sno_a = np.full((len(manifolds), len(manifolds)), np.nan, dtype=manifolds[0].centroid.dtype)
    sno_b = np.full((len(manifolds), len(manifolds)), np.nan, dtype=manifolds[0].centroid.dtype)
    for i in range(len(manifolds) - 1):
        for j in range(i + 1, len(manifolds)):
            man_i, man_j = manifolds[i], manifolds[j]
            sigdir_i = (man_i.centroid - man_j.centroid) / man_i.radius
            sigdir_j = (man_j.centroid - man_i.centroid) / man_j.radius
            sno_a[i, j] = np.linalg.norm(man_i.directions_of_variance @ sigdir_i)
            sno_a[j, i] = np.linalg.norm(man_j.directions_of_variance @ sigdir_j)
            sno_b[i, j] = np.linalg.norm(man_j.directions_of_variance @ sigdir_i)
            sno_b[j, i] = np.linalg.norm(man_i.directions_of_variance @ sigdir_j)
    return sno_a, sno_b


def manifold_signal_noise_ratio(signal: np.ndarray, bias: np.ndarray, dimensionality: np.ndarray,
                                sno_a: np.ndarray, sno_b: np.ndarray, num_examples: int) -> np.ndarray:
    numerator = signal ** 2 + ((bias + 1) ** 2 - 1) / num_examples
    denominator = np.sqrt(1 / (dimensionality[:, np.newaxis] * num_examples) +
                          sno_b ** 2 / num_examples + sno_a ** 2)
    return 0.5 * numerator / denominator


def manifold_global_statistics(manifolds: List[ManifoldGeometry]) -> Tuple[float, float]:
    pca = PCA()
    pca.fit(np.stack([m.centroid for m in manifolds]))
    global_radius = np.sqrt(pca.explained_variance_.mean())
    global_dimensionlity = pca.explained_variance_.sum() ** 2 / (pca.explained_variance_ ** 2).sum()
    return global_radius, global_dimensionlity


def statistics(X: np.ndarray) -> Tuple[float, float]:
    return np.nanmean(X).item(), np.nanstd(X).item()


def test_manifold_geometry():
    from matplotlib import pyplot as plt
    import seaborn as sns
    np.random.seed(27)
    sns.set(style='darkgrid')

    mean_a, mean_b = np.array([-1, -1]), \
                     np.array([1, 1])
    eigvals_a, eigvals_b = np.array([0.2, 0.05]), \
                           np.array([0.1, 0.1])
    eigvecs_a, eigvecs_b = np.array([[-np.sqrt(.5), np.sqrt(.5)], [np.sqrt(.5), np.sqrt(.5)]]), \
                           np.array([[0, 1], [1, 0]])
    cov_a, cov_b = eigvecs_a @ np.diag(eigvals_a) @ eigvecs_a.T, \
                   eigvecs_b @ np.diag(eigvals_b) @ eigvecs_b.T
    X_a, X_b = np.random.multivariate_normal(mean_a, cov_a, 500), \
               np.random.multivariate_normal(mean_b, cov_b, 500)
    plt.scatter(X_a[:, 0], X_a[:, 1], alpha=0.3, label='Concept A')
    plt.scatter(X_b[:, 0], X_b[:, 1], alpha=0.3, label='Concept B')
    plt.axis('square')
    plt.legend()

    man_a, man_b = ManifoldGeometry(X_a), ManifoldGeometry(X_b)
    rad = manifold_radius([man_a, man_b])
    signal = manifold_signal([man_a, man_b])
    bias = manifold_bias([man_a, man_b])
    dim = manifold_dimensionality([man_a, man_b])
    sno_a, sno_b = manifold_signal_noise_overlap([man_a, man_b])
    snr = manifold_signal_noise_ratio(signal, bias, dim, sno_a, sno_b, man_a.num_examples)
    text_a = 'Concept A\n' \
             f'Radius:    {rad[0]:.3g}\n' \
             f'Signal:    {signal[0, 1]:.3g}\n' \
             f'Bias:      {bias[0, 1]:.3g}\n' \
             f'Dimension: {dim[0]:.3g}\n' \
             f'Self-SNO:  {sno_a[0, 1]:.3g}\n' \
             f'Other-SNO: {sno_b[0, 1]:.3g}\n' \
             f'SNR:       {snr[0, 1]:.3g}'
    plt.figtext(1.1, 0.9, text_a,
                fontsize=12, fontfamily='monospace', va='top',
                transform=plt.gca().transAxes)
    text_b = 'Concept B\n' \
             f'Radius:    {rad[1]:.3g}\n' \
             f'Signal:    {signal[1, 0]:.3g}\n' \
             f'Bias:      {bias[1, 0]:.3g}\n' \
             f'Dimension: {dim[1]:.3g}\n' \
             f'Self-SNO:  {sno_a[1, 0]:.3g}\n' \
             f'Other-SNO: {sno_b[1, 0]:.3g}\n' \
             f'SNR:       {snr[1, 0]:.3g}'
    plt.figtext(1.1, 0.4, text_b,
                fontsize=12, fontfamily='monospace', va='top',
                transform=plt.gca().transAxes)

    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    plt.show()