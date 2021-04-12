# -*- coding: utf-8 -*-
from typing import Sequence, Union

import numpy as np
import scipy

from stonesoup.measures import ObservationAccuracy
from .base import MeasurementModel
from ...base import Property
from ...models.base import ReversibleModel
from ...types.array import Matrix, StateVector, StateVectors


class BasicTimeInvariantObservervation(MeasurementModel, ReversibleModel):
    emission_matrix: Matrix = Property(
        doc=r"Matrix defining emissions from measurement classes. In essence, it defines the "
            r"probability an observed target is a particular hidden class :math:`\phi_{i}`, given "
            r"it has been observed to be measured class :math:`z_{j}`. "
            r":math:`E_{ij} = P(\phi_{i} | z_{j})`.")
    reverse_emission: Matrix = Property(
        default=None,
        doc=r"Matrix utilised in generating observations. Defines the probability a target of "
            r"hidden class :math:`\phi_{j}` will be observed as measurement class :math:`z_{i}`. "
            r":math:`K_{ij} = P(z_{i} | \phi_{j})`")
    mapping: Sequence = Property(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Normalise emission rows
        self.emission_matrix = \
            self.emission_matrix / self.emission_matrix.sum(axis=0)[np.newaxis, :]

        # Default reverse emission is normalised emission transpose
        if self.reverse_emission is None:
            self.reverse_emission = \
                self.emission_matrix.T / self.emission_matrix.T.sum(axis=0)[np.newaxis, :]

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of observation dimensions
        """

        return np.shape(self.reverse_emission)[0]

    @property
    def mapping(self):
        return range(np.shape(self.emission)[0])

    def function(self, state, **kwargs):
        """Observer function :math:`HX_{t}`

        Parameters
        ----------
        state: :class:`~.State`
            An input (hidden class) state

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The observer function evaluated and resultant categorical distribution sampled from to
            represent a determinate measurement class.
        """

        y = self.reverse_emission @ state.state_vector

        y = y / np.sum(y)

        sample = self._sample(y.flatten())

        return StateVector(sample)

    def inverse_function(self, detection, **kwargs) -> StateVector:
        return self.emission_matrix @ detection.state_vector

    def jacobian(self, state, **kwargs):
        raise NotImplementedError("Jacobian for observation measurement model is not defined.")

    def _sample(self, row):
        rv = scipy.stats.multinomial(n=1, p=row)
        return rv.rvs(size=1, random_state=None)

    @staticmethod
    def measure():
        return ObservationAccuracy()

    def pdf(self, state1, state2, **kwargs):
        return self.measure(state1, state2)

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        raise NotImplementedError("Noise generation for observation-based measurements is not "
                                  "implemented")