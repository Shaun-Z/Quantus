"""This module contains constants and simple methods to retreive the available metrics, perturbation-,
similarity-, normalisation- functions and explanation methods in Quantus."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import sys
from typing import List, Dict, Mapping, Type
from quantus.functions import n_bins_func, perturb_func, similarity_func, normalise_func, loss_func, norm_func
from quantus.metrics import *

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final


AVAILABLE_METRICS: Final[Mapping[str, Mapping[str, Type[Metric]]]] = {
    "Faithfulness": {
        "Faithfulness Correlation": FaithfulnessCorrelation,
        "Faithfulness Estimate": FaithfulnessEstimate,
        "Pixel-Flipping": PixelFlipping,
        "Region Segmentation": RegionPerturbation,
        "Monotonicity-Arya": Monotonicity,
        "Monotonicity-Nguyen": MonotonicityCorrelation,
        "Selectivity": Selectivity,
        "SensitivityN": SensitivityN,
        "IROF": IROF,
        "ROAD": ROAD,
        "Infidelity": Infidelity,
        "Sufficiency": Sufficiency,
    },
    "Robustness": {
        "Continuity Test": Continuity,
        "Local Lipschitz Estimate": LocalLipschitzEstimate,
        "Max-Sensitivity": MaxSensitivity,
        "Avg-Sensitivity": AvgSensitivity,
        "Consistency": Consistency,
        "Relative Input Stability": RelativeInputStability,
        "Relative Output Stability": RelativeOutputStability,
        "Relative Representation Stability": RelativeRepresentationStability,
    },
    "Localisation": {
        "Pointing Game": PointingGame,
        "Top-K Intersection": TopKIntersection,
        "Relevance Mass Accuracy": RelevanceMassAccuracy,
        "Relevance Rank Accuracy": RelevanceRankAccuracy,
        "Attribution Localisation ": AttributionLocalisation,
        "AUC": AUC,
        "Focus": Focus,
    },
    "Complexity": {
        "Sparseness": Sparseness,
        "Complexity": Complexity,
        "Effective Complexity": EffectiveComplexity,
    },
    "Randomisation": {
        "MPRT": MPRT,
        "Smooth MPRT": SmoothMPRT,
        "Efficient MPRT": EfficientMPRT,
        "Random Logit": RandomLogit,
    },
    "Axiomatic": {
        "Completeness": Completeness,
        "NonSensitivity": NonSensitivity,
        "InputInvariance": InputInvariance,
    },
}


AVAILABLE_PERTURBATION_FUNCTIONS = {
    "baseline_replacement_by_indices": perturb_func.baseline_replacement_by_indices,
    "baseline_replacement_by_shift": perturb_func.baseline_replacement_by_shift,
    "baseline_replacement_by_blur": perturb_func.baseline_replacement_by_blur,
    "gaussian_noise": perturb_func.gaussian_noise,
    "uniform_noise": perturb_func.uniform_noise,
    "rotation": perturb_func.rotation,
    "translation_x_direction": perturb_func.translation_x_direction,
    "translation_y_direction": perturb_func.translation_y_direction,
    "no_perturbation": perturb_func.no_perturbation,
    "noisy_linear_imputation": perturb_func.noisy_linear_imputation,
}


AVAILABLE_SIMILARITY_FUNCTIONS = {
    "correlation_spearman": similarity_func.correlation_spearman,
    "correlation_pearson": similarity_func.correlation_pearson,
    "correlation_kendall_tau": similarity_func.correlation_kendall_tau,
    "distance_euclidean": similarity_func.distance_euclidean,
    "distance_manhattan": similarity_func.distance_manhattan,
    "distance_chebyshev": similarity_func.distance_chebyshev,
    "lipschitz_constant": similarity_func.lipschitz_constant,
    "abs_difference": similarity_func.abs_difference,
    "squared_difference": similarity_func.squared_difference,
    "difference": similarity_func.difference,
    "cosine": similarity_func.cosine,
    "ssim": similarity_func.ssim,
    "mse": loss_func.mse,
}

AVAILABLE_NORMALISATION_FUNCTIONS = {
    "normalise_by_negative": normalise_func.normalise_by_negative,
    "normalise_by_max": normalise_func.normalise_by_max,
    "denormalise": normalise_func.denormalise,
}


AVAILABLE_XAI_METHODS_CAPTUM = [
    "GradientShap",
    "IntegratedGradients",
    "DeepLift",
    "DeepLiftShap",
    "InputXGradient",
    "Saliency",
    "FeatureAblation",
    "Deconvolution",
    "FeaturePermutation",
    "Lime",
    "KernelShap",
    "LRP",
    "Gradient",
    "Occlusion",
    "LayerGradCam",
    "GuidedGradCam",
    "LayerConductance",
    "LayerActivation",
    "InternalInfluence",
    "LayerGradientXActivation",
    "Control Var. Sobel Filter",
    "Control Var. Constant",
    "Control Var. Random Uniform",
]


DEPRECATED_XAI_METHODS_CAPTUM = {"GradCam": "LayerGradCam"}


AVAILABLE_XAI_METHODS_TF = [
    "VanillaGradients",
    "IntegratedGradients",
    "GradientsInput",
    "OcclusionSensitivity",
    "GradCAM",
    "SmoothGrad",
]


DEPRECATED_XAI_METHODS_TF = {
    "Gradient": "VanillaGradients",
    "InputXGradient": "GradientsInput",
    "Occlusion": "OcclusionSensitivity",
    "GradCam": "GradCAM",
}


AVAILABLE_N_BINS_ALGORITHMS = {
    "Freedman Diaconis": n_bins_func.freedman_diaconis_rule,
    "Scotts": n_bins_func.scotts_rule,
    "Square Root": n_bins_func.square_root_choice,
    "Sturges Formula": n_bins_func.sturges_formula,
    "Rice": n_bins_func.rice_rule,
}


def available_categories() -> List[str]:
    """
    Retrieve the available metric categories in Quantus.

    Returns
    -------
    List[str]
        With the available metric categories in Quantus.
    """
    return [c for c in AVAILABLE_METRICS.keys()]


def available_metrics() -> Dict[str, List[str]]:
    """
    Retrieve the available metrics in Quantus.

    Returns
    -------
    Dict[str, str]
        With the available metrics, under each category in Quantus.
    """
    return {c: list(metrics.keys()) for c, metrics in AVAILABLE_METRICS.items()}


def available_methods_tf_explain() -> List[str]:
    """
    Retrieve the available explanation methods in Quantus.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_XAI_METHODS_TF]


def available_methods_captum() -> List[str]:
    """
    Retrieve the available explanation methods in Quantus.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_XAI_METHODS_CAPTUM]


def available_perturbation_functions() -> List[str]:
    """
    Retrieve the available perturbation functions in Quantus.

    Returns
    -------
    List[str]
        With the available perturbation functions in Quantus.
    """
    return [c for c in AVAILABLE_PERTURBATION_FUNCTIONS.keys()]


def available_similarity_functions() -> List[str]:
    """
    Retrieve the available similarity functions in Quantus.

    Returns
    -------
    List[str]
        With the available similarity functions in Quantus.
    """
    return [c for c in AVAILABLE_SIMILARITY_FUNCTIONS.keys()]


def available_normalisation_functions() -> List[str]:
    """
    Retrieve the available normalisation functions in Quantus.

    Returns
    -------
    List[str]
        With the available normalisation functions in Quantus.
    """
    return [c for c in AVAILABLE_NORMALISATION_FUNCTIONS.keys()]
