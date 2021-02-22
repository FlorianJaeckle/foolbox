from typing import Union, Tuple, Any, Optional
from functools import partial
import numpy as np
import eagerpy as ep
# import torch
import time

from ..devutils import flatten
from ..devutils import atleast_kd

from ..types import Bounds

from ..models import Model

from ..distances import l2
from ..distances import linf

from ..criteria import Misclassification
from ..criteria import TargetedMisclassification

from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs


class L2CarliniWagnerAttack(MinimizationAttack):
    """Implementation of the Carlini & Wagner L2 Attack. [#Carl16]_

    Args:
        binary_search_steps : Number of steps to perform in the binary search
            over the const c.
        steps : Number of optimization steps within each binary search step.
        stepsize : Stepsize to update the examples.
        confidence : Confidence required for an example to be marked as adversarial.
            Controls the gap between example and decision boundary.
        initial_const : Initial value of the const c with which the binary search starts.
        abort_early : Stop inner search as soons as an adversarial example has been found.
            Does not affect the binary search over the const c.

    References:
        .. [#Carl16] Nicholas Carlini, David Wagner, "Towards evaluating the robustness of
            neural networks. In 2017 ieee symposium on security and privacy"
            https://arxiv.org/abs/1608.04644
    """

    distance = l2

    def __init__(
        self,
        binary_search_steps: int = 9,
        steps: int = 10000,
        stepsize: float = 1e-2,
        confidence: float = 0,
        initial_const: float = 1e-3,
        abort_early: bool = True,
    ):
        self.binary_search_steps = binary_search_steps
        self.steps = steps
        self.stepsize = stepsize
        self.confidence = confidence
        self.initial_const = initial_const
        self.abort_early = abort_early

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
            change_classes_logits = -self.confidence
        else:
            raise ValueError("unsupported criterion")

        def is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            if change_classes_logits != 0:
                logits += ep.onehot_like(logits, classes, value=change_classes_logits)
            return criterion_(perturbed, logits)

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        bounds = model.bounds
        to_attack_space = partial(_to_attack_space, bounds=bounds)
        to_model_space = partial(_to_model_space, bounds=bounds)

        x_attack = to_attack_space(x)
        reconstsructed_x = to_model_space(x_attack)

        rows = range(N)

        def loss_fun(
            delta: ep.Tensor, consts: ep.Tensor
        ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            assert delta.shape == x_attack.shape
            assert consts.shape == (N,)

            x = to_model_space(x_attack + delta)
            logits = model(x)

            if targeted:
                c_minimize = best_other_classes(logits, classes)
                c_maximize = classes  # target_classes
            else:
                c_minimize = classes  # labels
                c_maximize = best_other_classes(logits, classes)

            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)

            is_adv_loss = is_adv_loss + self.confidence
            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts

            squared_norms = flatten(x - reconstsructed_x).square().sum(axis=-1)
            loss = is_adv_loss.sum() + squared_norms.sum()
            return loss, (x, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        consts = self.initial_const * np.ones((N,))
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.full(x, (N,), ep.inf)

        # the binary search searches for the smallest consts that produce adversarials
        for binary_search_step in range(self.binary_search_steps):
            if (
                binary_search_step == self.binary_search_steps - 1
                and self.binary_search_steps >= 10
            ):
                # in the last binary search step, repeat the search once
                consts = np.minimum(upper_bounds, 1e10)

            # create a new optimizer find the delta that minimizes the loss
            delta = ep.zeros_like(x_attack)
            optimizer = AdamOptimizer(delta)

            # tracks whether adv with the current consts was found
            found_advs = np.full((N,), fill_value=False)
            loss_at_previous_check = np.inf

            consts_ = ep.from_numpy(x, consts.astype(np.float32))

            for step in range(self.steps):
                loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_)
                delta += optimizer(gradient, self.stepsize)

                if self.abort_early and step % (np.ceil(self.steps / 10)) == 0:
                    # after each tenth of the overall steps, check progress
                    if not (loss <= 0.9999 * loss_at_previous_check):
                        break  # stop Adam if there has been no progress
                    loss_at_previous_check = loss

                found_advs_iter = is_adversarial(perturbed, logits)
                found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

                norms = flatten(perturbed - x).norms.l2(axis=-1)
                closer = norms < best_advs_norms
                new_best = ep.logical_and(closer, found_advs_iter)

                new_best_ = atleast_kd(new_best, best_advs.ndim)
                best_advs = ep.where(new_best_, perturbed, best_advs)
                best_advs_norms = ep.where(new_best, norms, best_advs_norms)

            upper_bounds = np.where(found_advs, consts, upper_bounds)
            lower_bounds = np.where(found_advs, lower_bounds, consts)

            consts_exponential_search = consts * 10
            consts_binary_search = (lower_bounds + upper_bounds) / 2
            consts = np.where(
                np.isinf(upper_bounds), consts_exponential_search, consts_binary_search
            )

        return restore_type(best_advs)


class LinfCarliniWagnerAttack(MinimizationAttack):
    # NOTE: WORK IN PROGRESS, DO NOT USE YET
    """Implementation of the Carlini & Wagner Ling Attack. [#Carl16]_

    Args:
        steps : Number of optimization steps within each binary search step.
        stepsize : Stepsize to update the examples.
        initial_const : Initial value of the const c with which the binary search starts.
        largest_const : the largest value of c to go up to before giving up
        abort_early : Stop inner search as soons as an adversarial example has been found.
            Does not affect the binary search over the const c.
        decrease_factor: 0<f<1, rate at which we shrink tau; larger is more accurate
        reduce_const: try to lower c each iteration; faster to set to false
        const_factor : f>1, rate at which we increase constant, smaller better


    References:
        .. [#Carl16] Nicholas Carlini, David Wagner, "Towards evaluating the robustness of
            neural networks. In 2017 ieee symposium on security and privacy"
            https://arxiv.org/abs/1608.04644
    """
    # check whether any more changes need to be done for l infinity
    # official python code: https://github.com/carlini/nn_robust_attacks/blob/master/li_attack.py
    # the current implementation doesn't work because taking the gradient at linf is not very helpful
    # support early stopping as soon as our eps is less than the target
    # maybe add random initialization rather than setting delta to all zeros

    distance = linf

    def __init__(
        self,
        steps: int = 10000,
        stepsize: float = 1e-3,
        initial_const: float = 1e-5,
        largest_const: float = 2e+1,
        abort_early: bool = True,
        decrease_factor: float = 0.9,
        reduce_const: bool = False,
        const_factor: float = 2.0,
    ):
        self.steps = steps
        self.stepsize = stepsize
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.abort_early = abort_early
        self.decrease_factor = decrease_factor
        self.reduce_const = reduce_const
        self.const_factor = const_factor
        self.warm_start = True

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        if 'eps' in kwargs.keys():
            eps_needed = kwargs['eps']
        else:
            eps_needed = None
        # raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
            change_classes_logits = self.confidence
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        def is_adversarial(perturbed: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            return criterion_(perturbed, logits)

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        bounds = model.bounds
        to_attack_space = partial(_to_attack_space, bounds=bounds)
        to_model_space = partial(_to_model_space, bounds=bounds)

        x_attack = to_attack_space(x)
        reconstsructed_x = to_model_space(x_attack)

        rows = range(N)

        def loss_fun(
            delta: ep.Tensor, consts: ep.Tensor, tau: ep.Tensor
        ) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            assert delta.shape == x_attack.shape

            x = to_model_space(x_attack + delta)
            logits = model(x)

            if targeted:
                c_minimize = best_other_classes(logits, classes)
                c_maximize = classes  # target_classes
            else:
                c_minimize = classes  # labels
                c_maximize = best_other_classes(logits, classes)

            is_adv_loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert is_adv_loss.shape == (N,)

            is_adv_loss = ep.maximum(0, is_adv_loss)
            is_adv_loss = is_adv_loss * consts

            linf_norms = ep.clip((flatten(x - reconstsructed_x).abs() - tau), min_=0, max_=None).sum(axis=1)
            loss = is_adv_loss.sum() + linf_norms.sum()
            return loss, (x, logits)

        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        const = self.initial_const
        lower_bounds = np.zeros((N,))
        upper_bounds = np.inf * np.ones((N,))

        best_advs = ep.zeros_like(x)
        best_advs_norms = ep.full(x, (N,), ep.inf)

        # perturbation initialized to all zeros
        delta = ep.zeros_like(x_attack)

        tau = 1.0
        timeout = 100
        time_start = time.time()

        # we gradually reduce tau
        while tau > 1./10 and time.time()-time_start < timeout:  # in the original code this was `while tau > 1./256` but seems pointless for our case to decrease tau that much
            # try to solve given this tau value
            # print(f"tau: {tau}, const: {const}")

            succ = False  # flag indicating whether the current attack was successful
            # the binary search searches for the smallest consts that produce adversarials
            while const < self.largest_const and time.time()-time_start < timeout:
                if not self.warm_start:
                    # initialized to all zeros
                    delta = ep.zeros_like(x_attack)
                # create a new optimizer find the delta that minimizes the loss --- maybe warm start the optimizer as well?
                optimizer = AdamOptimizer(delta)

                # tracks whether adv with the current consts was found
                found_advs = np.full((N,), fill_value=False)
                loss_at_previous_check = np.inf

                # consts_ = ep.from_numpy(x, const.astype(np.float32))
                consts_ = const

                # print('consts', consts_, end="\r")

                for step in range(self.steps):
                    # delta is the current perturbation - we call loss_aux_and_grad to get a new gradient
                    # as well as the current new image and loss (from a pytorch package instead of autograd)
                    loss, (perturbed, logits), gradient = loss_aux_and_grad(delta, consts_, tau)
                    # we update the current perturbation using the gradient and the adam optimizer
                    delta += optimizer(gradient, self.stepsize)

                    found_advs_iter = is_adversarial(perturbed, logits)
                    found_advs = np.logical_or(found_advs, found_advs_iter.numpy())

                    norms = flatten(perturbed - x).norms.linf(axis=-1)

                    closer = norms < best_advs_norms
                    new_best = ep.logical_and(closer, found_advs_iter)

                    new_best_ = atleast_kd(new_best, best_advs.ndim)
                    best_advs = ep.where(new_best_, perturbed, best_advs)
                    best_advs_norms = ep.where(new_best, norms, best_advs_norms)

                    if self.abort_early and loss < 0.0001*const:
                        works = is_adversarial(perturbed, logits)
                        if ep.min(works):
                            # the attack for the given tau worked
                            succ = True
                            break

                if succ:
                    break

                upper_bounds = np.where(found_advs, const, upper_bounds)
                lower_bounds = np.where(found_advs, lower_bounds, const)

                # we didn't succeed, increase constant and try again
                const *= self.const_factor

            # print("best_advs", best_advs_norms)

            if not succ:
                # the last attack failed so we return our latest answer
                break


            if eps_needed:
                # print(best_advs_norms, eps_needed)
                if best_advs_norms < eps_needed:
                    print("\nsuccess in CW\n")
                    return restore_type(best_advs)
                    break

            # the attack succeeded, reduce tau and try again

            if self.reduce_const:
                const = const/2

            actualtau = norms.max()

            if actualtau < tau:
                tau = actualtau

            # TODO warm start grad
            # prev = nimg

            tau *= self.decrease_factor

        print("best_advs", best_advs_norms)
        return restore_type(best_advs)


class AdamOptimizer:
    def __init__(self, x: ep.Tensor):
        self.m = ep.zeros_like(x)
        self.v = ep.zeros_like(x)
        self.t = 0

    def __call__(
        self,
        gradient: ep.Tensor,
        stepsize: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> ep.Tensor:
        self.t += 1

        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * gradient ** 2

        bias_correction_1 = 1 - beta1 ** self.t
        bias_correction_2 = 1 - beta2 ** self.t

        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2

        return -stepsize * m_hat / (ep.sqrt(v_hat) + epsilon)


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)


def _to_attack_space(x: ep.Tensor, *, bounds: Bounds) -> ep.Tensor:
    min_, max_ = bounds
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = (x - a) / b  # map from [min_, max_] to [-1, +1]
    x = x * 0.999999  # from [-1, +1] to approx. (-1, +1)
    x = x.arctanh()  # from (-1, +1) to (-inf, +inf)
    return x


def _to_model_space(x: ep.Tensor, *, bounds: Bounds) -> ep.Tensor:
    min_, max_ = bounds
    x = x.tanh()  # from (-inf, +inf) to (-1, +1)
    a = (min_ + max_) / 2
    b = (max_ - min_) / 2
    x = x * b + a  # map from (-1, +1) to (min_, max_)
    return x
