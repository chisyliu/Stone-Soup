# -*- coding: utf-8 -*-
from typing import Sequence

import numpy as np
import scipy

from .base import MeasurementModel
from ...base import Property
from ...models.base import ReversibleModel
from ...types.array import Matrix, StateVector


class BasicTimeInvariantObservervation(MeasurementModel, ReversibleModel):
    emission_matrix: Matrix = Property(
        doc="Matrix defining emissions from measurement classes. In essence, it defines the "
            "probability an observed target is a particular hidden class :math:`\phi_{i}`, given "
            "it has been observed to be measured class :math:`z_{j}`. "
            ":math:`E_{ij} = P(\phi_{i} | z_{j}).")
    reverse_emission: Matrix = Property(
        default=None,
        doc="Matrix utilised in generating observations. Defines the probability a target of "
            "hidden class :math:`\phi_{j} will be observed as measuremnt class :math:`z_{i}`. "
            ":math:`K_{ij} = P(z_{i} | \phi_{j})`")
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
            The observer function evaluated and resultant categorical distribution sampled from.
        """

        y = self.reverse_emission @ state.state_vector

        y = y / np.sum(y)

        sample = self._sample(y.flatten())

        return StateVector(sample)

    def inverse_function(self, detection, **kwargs) -> StateVector:
        return self.emission_matrix @ detection.state_vector

    def _sample(self, row):
        rv = scipy.stats.multinomial(n=1, p=row)
        return rv.rvs(size=1, random_state=None)

    def rvs(self):
        pass

    def pdf(self):
        pass
