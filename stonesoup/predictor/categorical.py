# -*- coding: utf-8 -*-
from ..base import Property
from ..models.transition import TransitionModel
from ..predictor import Predictor
from ..predictor._utils import predict_lru_cache
from ..types.prediction import CategoricalStatePrediction


class HMMPredictor(Predictor):
    r"""Models the prediction step of a hidden markov model"""

    transition_model: TransitionModel = Property(doc="The transition model to be used. This "
                                                     "should be a categorical transition model.")

    def _transition_function(self, prior, **kwargs):
        return self.transition_model.function(prior, **kwargs)

    def _predict_over_interval(self, prior, timestamp):
        """Private function to get the prediction interval (or None)

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state

        timestamp : :class:`datetime.datetime`, optional
            The (current) timestamp

        Returns
        -------
        : :class:`datetime.timedelta`
            time interval to predict over

        """

        # Deal with undefined timestamps
        if timestamp is None or prior.timestamp is None:
            predict_over_interval = None
        else:
            predict_over_interval = timestamp - prior.timestamp

        return predict_over_interval

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""A simple matrix multiplication. The Chapman-Kolmogorov equation is:

        .. math::
            p(x_k|z_{1:k-1}) &= \Sigma_{x_{k-1}} p(x_k|x_{k-1}) p(x_{k-1}|z_{1:k-1})\\
                             &= F_k p(x_{k-1}|z_{1:k-1})

        where :math:`F_k` is the category-transition matrix and :math:`p(x)` is encoded in the
        state vector

        Parameters
        ----------
        prior : :class:`~.CategoricalState`
            :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`, optional
            :math:`k + 1`
        **kwargs :
            These are passed to the :meth:`transition_model.function` method.

        Returns
        -------
        : :class:`~.StatePrediction`
            :math:`\mathbf{x}_{t + \Delta t|t}`, the predicted state.
        """
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        x_pred = self._transition_function(prior, time_interval=predict_over_interval, **kwargs)

        return CategoricalStatePrediction(state_vector=x_pred,
                                          timestamp=timestamp,
                                          num_categories=prior.num_categories,
                                          category_names=prior.category_names)
