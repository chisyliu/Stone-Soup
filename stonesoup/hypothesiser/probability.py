from scipy.stats import multivariate_normal as mn

from .base import Hypothesiser
from ..base import Property
from ..measures import Measure, ObservationAccuracy
from ..predictor import Predictor
from ..predictor.classification import ClassificationPredictor
from ..types.detection import MissedDetection
from ..types.hypothesis import SingleProbabilityHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
from ..updater import Updater
from ..updater.classification import ClassificationUpdater


class PDAHypothesiser(Hypothesiser):
    """Hypothesiser based on Probabilistic Data Association (PDA)

    Generate track predictions at detection times and calculate probabilities
    for all prediction-detection pairs for single prediction and multiple
    detections.
    """

    predictor: Predictor = Property(doc="Predict tracks to detection times")
    updater: Updater = Property(doc="Updater used to get measurement prediction")
    clutter_spatial_density: float = Property(
        doc="Spatial density of clutter - tied to probability of false detection")
    prob_detect: Probability = Property(
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_gate: Probability = Property(
        default=Probability(0.95),
        doc="Gate Probability - prob. gate contains true measurement "
            "if detected")

    def hypothesise(self, track, detections, timestamp):
        r"""Evaluate and return all track association hypotheses.

        For a given track and a set of N detections, return a
        MultipleHypothesis with N+1 detections (first detection is
        a 'MissedDetection'), each with an associated probability.
        Probabilities are assumed to be exhaustive (sum to 1) and mutually
        exclusive (two detections cannot be the correct association at the
        same time).

        Detection 0: missed detection, none of the detections are associated
        with the track.
        Detection :math:`i, i \in {1...N}`: detection i is associated
        with the track.

        The probabilities for these detections are calculated as follow:

        .. math::

          \beta_i(k) = \begin{cases}
                \frac{\mathcal{L}_{i}(k)}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=1,...,m(k) \\
                \frac{1-P_{D}P_{G}}{1-P_{D}P_{G}+\sum_{j=1}^{m(k)}
                  \mathcal{L}_{j}(k)}, \quad i=0
                \end{cases}

        where

        .. math::

          \mathcal{L}_{i}(k) = \frac{\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),
          S(k)]P_{D}}{\lambda}

        :math:`\lambda` is the clutter density

        :math:`P_{D}` is the detection probability

        :math:`P_{G}` is the gate probability

        :math:`\mathcal{N}[z_{i}(k);\hat{z}(k|k-1),S(k)]` is the likelihood
        ratio of the measurement :math:`z_{i}(k)` originating from the track
        target rather than the clutter.

        NOTE: Since all probabilities have the same denominator and are
        normalized later, the denominator can be discarded.

        References:

        [1] "The Probabilistic Data Association Filter: Estimation in the
        Presence of Measurement Origin Uncertainty" -
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5338565

        [2] "Robotics 2 Data Association" (Lecture notes) -
        http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-15-dataassociation.pdf

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on
        detections: :class:`list`
            A list of :class:`~Detection` objects, representing the available
            detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement
            predictions. Note that if a given detection has a non empty
            timestamp, then prediction will be performed according to
            the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleProbabilityHypothesis` objects

        """

        hypotheses = list()

        # Common state & measurement prediction
        prediction = self.predictor.predict(track, timestamp=timestamp)
        # Missed detection hypothesis
        probability = Probability(1 - self.prob_detect*self.prob_gate)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
                ))

        # True detection hypotheses
        for detection in detections:
            # Re-evaluate prediction
            prediction = self.predictor.predict(
                track, timestamp=detection.timestamp)
            # Compute measurement prediction and probability measure
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model)
            # Calculate difference before to handle custom types (mean defaults to zero)
            # This is required as log pdf coverts arrays to floats
            log_pdf = mn.logpdf(
                (detection.state_vector - measurement_prediction.state_vector).ravel(),
                cov=measurement_prediction.covar)
            pdf = Probability(log_pdf, log_value=True)
            probability = (pdf * self.prob_detect)/self.clutter_spatial_density

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction))

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)


class ClassificationHypothesiser(Hypothesiser):
    r"""Hypothesiser based on the consideration of multinomial distribution accuracy.
    Whereby it is assumed track space is a finite space of discrete classifications
    :math:`\{\phi_i | i \in \mathbb{Z}_{>0}\}` and measurement spaces are finite spaces of
    discrete measurement classifications :math:`\{\y_i | i \in \mathbb{Z}_{>0}\}`, whereby a
    state or measurement vector describes a multinomial distribution in its corresponding space,
    with each element :math:`i` describing the probability that the owning object is of
    classification :math:`\phi_i` or :math:`\y_i` respectively.

    This hypothesiser generates track predictions at detection times and scores each hypothesised
    prediction-detection pair according to the accuracy of the prediction to the detection's state
    space emission, calculated as the product of the detection's measurement model's emission
    matrix and detection vector product :math:`EZ` (which describes a multinomial distribution in
    the state space. This uses the :class:`~.ObservationAccuracy' measure by default, which gives
    a higher probability/score the closer the two distributions (prediction and emission) are to
    each other.
    """

    predictor: ClassificationPredictor = Property(doc="Predict tracks to detection times")
    updater: ClassificationUpdater = Property(doc="Updater used to get measurement prediction")
    clutter_spatial_density: float = Property(
        doc="Spatial density of clutter - tied to probability of false detection")
    prob_detect: Probability = Property(
        default=Probability(0.85),
        doc="Target Detection Probability")
    prob_gate: Probability = Property(
        default=Probability(0.95),
        doc="Gate Probability - prob. gate contains true measurement if detected")
    measure: Measure = Property(
        default=ObservationAccuracy,
        doc="Measure type to determine accuracy of prediction-measurement pairs")

    def hypothesise(self, track, detections, timestamp):
        """ Evaluate and return all track association hypotheses.

        For a given track and a set of N available detections, return a MultipleHypothesis object
        with N+1 detections (first detection is a 'MissedDetection'), each with an associated
        accuracy (of prediction to measurement emission) measure.

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on.
        detections: :class:`list`
            A list of :class:`~Detection` objects, representing the available detections.
        timestamp: :class:`datetime.datetime`
            A timestamp used when evaluating the state and measurement predictions. Note that if a
            given detection has a non empty timestamp, then prediction will be performed according
            to the timestamp of the detection.

        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleProbabilityHypothesis` objects
        """

        hypotheses = list()

        prediction = self.predictor.predict(track, timestamp=timestamp)
        probability = Probability(1 - self.prob_detect * self.prob_gate)
        hypotheses.append(
            SingleProbabilityHypothesis(
                prediction,
                MissedDetection(timestamp=timestamp),
                probability
            ))

        for detection in detections:
            prediction = self.predictor.predict(
                track, timestamp=detection.timestamp)

            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model)

            measurement_model = detection.measurement_model
            emission = measurement_model.inverse_function(detection)

            pdf = self.measure(prediction, emission)
            probability = (pdf * self.prob_detect) / self.clutter_spatial_density

            # True detection hypothesis
            hypotheses.append(
                SingleProbabilityHypothesis(
                    prediction,
                    detection,
                    probability,
                    measurement_prediction))

        return MultipleHypothesis(hypotheses, normalise=True, total_weight=1)
