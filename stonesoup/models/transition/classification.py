# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from stonesoup.measures import ObservationAccuracy
from ...base import Property
from ...models.transition import TransitionModel
from ...types.array import Matrix, StateVector, StateVectors


class BasicTimeInvariantClassificationTransitionModel(TransitionModel):
    r"""Time invariant model of a classification transition

    The assumption is that an object can be classified as finitely many discrete classes
    :math:`\{\phi_{k}|k\in\Z^+\}`, with a state space defined by state vectors representing
    multinomial distributions over these classes :math:`\bar{x}_{{t}_{i}} = P(\phi_{i}, t)`,
    with constant probability :math:`P(\phi_i, t+1 | \phi_j, t)` of transitioning between these in
    any given time-step. This is modelled
    by the stochastic matrix :attr:`transition_matrix`, where the :math:`ij`-th element is given
    by :math:`P(\phi_i, t + \Delta t | \phi_j, t) \forall \Delta t > 0`.
    """
    transition_matrix: Matrix = Property(
        doc="Matrix :math:`F_{ij} = P(\phi^{i}_{t}|\phi^{j}_{t-1})` determining the probability "
            "that the state is class :math:`\phi^{j}` at time :math:`t` given that it was class "
            ":math:`\phi^{j}` at time :math:`t-1`.")
    transition_noise: Matrix = Property(
        default=None,
        doc="Matrix :math:`\omega_{ij}` defining additive noise to class transition. "
            "Noise added is given by :math:noise_{i} = \omega_{ij}F_{jk}x_{k}")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """

        return self.transition_matrix.shape[0]

    def function(self, state, noise: bool = False, **kwargs) -> StateVector:
        """Applies transformation :math:`f(\vec{\phi}_{t-1}, t) = F\vec{\phi}_{t-1} + noise` (note
        that this is then normalised) (under the assumption that a state vector defines a true
        object's categorical distribution).

        Parameters
        ----------
        state : :class:`stonesoup.state.State`
            The state to be transitioned according to the models in :py:attr:`~model_list`.

        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_state, 1`). The resultant state vector of the transition.
        """
        x = self.transition_matrix @ state.state_vector

        if noise:
            row = self.transition_noise @ x

            x = x + StateVector(row)

            x = x / np.sum(x)  # normalise

        return x

    def pdf(self, state1, state2, **kwargs):
        measure = ObservationAccuracy()
        return measure(state1, state2)

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        raise NotImplementedError("Noise generation for classification-based state transitions is "
                                  "not implemented")
