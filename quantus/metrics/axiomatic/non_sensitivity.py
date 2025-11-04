"""This module contains the implementation of the Non-Sensitivity metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# more details.
# You should have received a copy of the GNU Lesser General Public License
# along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL:
# <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import sys
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantus.functions.perturb_func import batch_baseline_replacement_by_indices
from quantus.helpers import asserts, warn
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)
from quantus.helpers.perturbation_utils import make_perturb_func
from quantus.metrics.base import Metric

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


@final
class NonSensitivity(Metric[List[float]]):
    """
    Implementation of NonSensitivity by Nguyen et al., 2020.

    Non-sensitivity measures if zero-importance is only assigned to features,
    that the model is not functionally dependent on.

    References:
        1) An-phi Nguyen and María Rodríguez Martínez.: "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
        2) Marco Ancona et al.: "Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley
        Values Approximation." ICML (2019): 272-281.
        3) Grégoire Montavon et al.: "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Non-Sensitivity"
    data_applicability = {DataType.IMAGE, DataType.TIMESERIES, DataType.TABULAR}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.LOWER
    evaluation_category = EvaluationCategory.AXIOMATIC

    def __init__(
        self,
        eps: float = 1e-5,
        features_in_step: int = 1,
        abs: bool = True,
        normalise: bool = True,
        normalise_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        perturb_baseline: str = "black",
        perturb_func: Optional[Callable] = None,
        perturb_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: bool = False,
        aggregate_func: Callable = np.mean,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        eps: float
            Attributions threshold, default=1e-5.
        features_in_step: integer
            The step size, default=1.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=True.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_baseline: string
            Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be produced over all instances.
        aggregate_func: callable
            A Callable to aggregate the scores per instance to one float.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        if perturb_func is None:
            perturb_func = batch_baseline_replacement_by_indices

        # Save metric-specific attributes.
        self.eps = eps
        self.features_in_step = features_in_step
        self.perturb_func = make_perturb_func(
            perturb_func, perturb_func_kwargs, perturb_baseline=perturb_baseline
        )

        # Asserts and warnings.
        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=(
                    "baseline value 'perturb_baseline', the number of samples to iterate"
                    " over 'n_samples' and the threshold value function for the feature"
                    " to be considered having an insignificant contribution to the model"
                ),
                citation=(
                    "Nguyen, An-phi, and María Rodríguez Martínez. 'On quantitative aspects of "
                    "model interpretability.' arXiv preprint arXiv:2007.07584 (2020)."
                ),
            )

    def __call__(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = None,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency)
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            softmax=softmax,
            device=device,
            model_predict_kwargs=model_predict_kwargs,
            batch_size=batch_size,
            **kwargs,
        )

    def custom_preprocess(
        self,
        x_batch: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        kwargs:
            Unused.

        Returns
        -------
        None
        """
        # Asserts.
        asserts.assert_features_in_step(
            features_in_step=self.features_in_step,
            input_shape=x_batch.shape[2:],
        )

    def evaluate_batch(
        self,
        model,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        **kwargs,
    ):
        """
        Evaluate a batch for the custom Non-Sensitivity metric.

        This implementation perturbs *feature* and *non-feature* pixels separately,
        evaluating how sensitive the model’s predictions are to each perturbation.
        The metric quantifies violations of the Non-Sensitivity principle in both directions:
        (1) when perturbing **non-feature** pixels *does* affect model predictions, and
        (2) when perturbing **feature** pixels *does not* affect model predictions.

        Parameters
        ----------
        model : object
            Model with `predict` and `shape_input` methods, compatible with Quantus conventions.
        x_batch : np.ndarray
            Input batch, of shape (B, C, H, W) or (B, H, W, C), depending on `channel_first`.
        y_batch : np.ndarray
            Ground truth or target class indices, of shape (B,).
        a_batch : np.ndarray
            Attribution maps aligned with `x_batch`, of shape (B, C, H, W).
        **kwargs :
            Additional keyword arguments passed through for flexibility.

        Returns
        -------
        np.ndarray
            Array of shape (B,), where each value indicates the total number of
            non-sensitivity violations per sample. Lower values indicate higher sensitivity
            (fewer violations), whereas higher values indicate non-sensitivity.

        Notes
        -----
        - The function assumes that a lower attribution value (below `self.eps`)
        represents a "non-feature" pixel.
        - Perturbations are applied in groups of size `self.features_in_step`.
        - The perturbation function `self.perturb_func` must follow the Quantus API:
        it receives an array and an index mask, and returns a perturbed copy.
        - Designed to comply with Quantus internal metric conventions and to be
        lint-clean under `black` and `flake8`.

        """
        # --- Step 1. Prepare shapes ---
        if x_batch.shape != a_batch.shape:
            a_batch = np.broadcast_to(a_batch, x_batch.shape)

        B = x_batch.shape[0]
        x_shape = x_batch.shape
        x_flat = x_batch.reshape(B, -1)
        a_flat = a_batch.reshape(B, -1)

        # --- Step 2. Split feature vs non-feature ---
        non_features = a_flat < self.eps
        features = ~non_features

        # --- Step 3. Get base predictions ---
        x_input = model.shape_input(x_batch, x_shape, channel_first=True, batched=True)
        y_pred = model.predict(x_input)[np.arange(B), y_batch]

        # --- Step 4. Allocate score map ---
        pixel_scores = np.zeros_like(a_flat, dtype=float)

        # --- Helper: perturbation loop ---
        def perturb_and_record(indices_mask, desc="nonfeature"):
            for b in range(B):
                indices = np.where(indices_mask[b])[0]
                n_pixels = len(indices)
                if n_pixels == 0:
                    continue

                n_steps = math.ceil(n_pixels / self.features_in_step)
                for step in range(n_steps):
                    print(
                        f"Processing batch {b+1}/{B}, {desc} step {step+1}/{n_steps}",
                        end="\r",
                    )
                    start = step * self.features_in_step
                    end = min((step + 1) * self.features_in_step, n_pixels)
                    subset_idx = indices[start:end]
                    indices_2d = np.expand_dims(subset_idx, axis=0)

                    # --- Perturb only selected pixels ---
                    perturbed_flat = x_flat.copy()
                    perturbed_flat[b] = self.perturb_func(
                        arr=perturbed_flat[b : b + 1, :],
                        indices=indices_2d,
                    )
                    x_perturbed = perturbed_flat.reshape(x_shape)
                    x_input = model.shape_input(
                        x_perturbed, x_shape, channel_first=True, batched=True
                    )
                    y_pred_perturb = model.predict(x_input)[np.arange(B), y_batch]

                    # Assign scores for the perturbed pixels
                    pixel_scores[b, subset_idx] = y_pred_perturb[b]

        # --- Step 5. Run loops ---
        perturb_and_record(non_features, "nonfeature")
        perturb_and_record(features, "feature")

        # --- Step 6. Reshape to image shape ---
        preds_differences = np.abs(y_pred[:, np.newaxis] - pixel_scores)
        preds_differences = preds_differences < self.eps
        pixel_scores = pixel_scores.reshape(x_shape)

        return (preds_differences ^ non_features).sum(-1)
