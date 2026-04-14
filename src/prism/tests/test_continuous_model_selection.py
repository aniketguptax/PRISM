import math

from prism.processes.linear_gaussian_ssm import LinearGaussianSSM
from prism.reconstruction.kalman_iss import KalmanISSReconstructor
from prism.representations.continuous import ISSDim


def _train_series(length: int = 520, split: int = 400, seed: int = 0):
    process = LinearGaussianSSM(a=0.9, process_std=0.3, obs_std=0.2, latent_dim=2, obs_dim=3)
    sample = process.sample(length=length, seed=seed)
    return sample.x[:split]


def test_hierarchical_macro_builder_runs() -> None:
    x_train = _train_series(length=520, split=400, seed=1)
    recon = KalmanISSReconstructor(
        em_iters=12,
        em_tol=1e-4,
        em_ridge=1e-6,
        macro_builder="hierarchical_complete",
        projection_mode="pca",
        macro_eps=0.25,
        macro_bins=3,
    )
    model = recon.fit(x_train, ISSDim(d=2, dv=2), seed=3)
    assert model.n_macro_states >= 1
    assert model.macro_builder == "hierarchical_complete"


def test_model_selection_predictive_returns_candidate_metadata() -> None:
    x_train = _train_series(length=520, split=400, seed=4)
    recon = KalmanISSReconstructor(
        em_iters=10,
        em_tol=1e-4,
        em_ridge=1e-6,
        model_select=True,
        selection_score="predictive",
        selection_train_frac=0.75,
        selection_projection_modes=("pca",),
        selection_macro_builders=("hierarchical_complete", "greedy"),
        selection_macro_eps=(0.25,),
        projection_mode="pca",
    )
    model = recon.fit(x_train, ISSDim(d=2, dv=2), seed=5)
    assert model.model_selection
    assert model.selection_score == "predictive"
    assert model.projection_mode == "pca"
    assert model.macro_builder in {"hierarchical_complete", "greedy"}
    assert math.isfinite(model.selection_predictive)


def test_model_selection_stability_returns_stability_score() -> None:
    x_train = _train_series(length=500, split=380, seed=8)
    recon = KalmanISSReconstructor(
        em_iters=8,
        em_tol=1e-4,
        em_ridge=1e-6,
        model_select=True,
        selection_score="stability",
        selection_train_frac=0.75,
        selection_projection_modes=("pca",),
        selection_macro_builders=("hierarchical_single", "greedy", "linear_quantile"),
        selection_macro_eps=(0.25,),
        selection_repeats=3,
        selection_perturb="seed_blocked",
        selection_block_frac=0.85,
        projection_mode="pca",
    )
    model = recon.fit(x_train, ISSDim(d=2, dv=2), seed=11)
    assert model.model_selection
    assert model.selection_score == "stability"
    assert math.isfinite(model.selection_stability)
    assert -1.0 <= model.selection_stability <= 1.0
    assert math.isfinite(model.selection_n_states_var)
    assert math.isfinite(model.selection_macro_time_mean_s)


def test_linear_quantile_baseline_runs() -> None:
    x_train = _train_series(length=520, split=400, seed=10)
    recon = KalmanISSReconstructor(
        em_iters=10,
        em_tol=1e-4,
        em_ridge=1e-6,
        macro_builder="linear_quantile",
        projection_mode="pca",
        macro_eps=0.2,
        macro_bins=3,
    )
    model = recon.fit(x_train, ISSDim(d=2, dv=2), seed=7)
    assert model.macro_builder == "linear_quantile"
    assert model.n_macro_states >= 1
    assert model.macro_distance_matrix_mb_est == 0.0
