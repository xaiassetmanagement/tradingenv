import tradingenv
from tradingenv.events import Observer
from sklearn.exceptions import NotFittedError
from datetime import datetime
from typing import Dict, Any
import numpy as np
import gymnasium.spaces
import copy


class Feature(Observer):
    """This class gives you the option to instance State as a collection of
    features. See docstring of State for further details.

    Attributes
    ----------
    exchange
        Instance of tradingenv.exchange.Exchange, storing current and
        historical asset prices.
    action_space
        Action space passed when instancing the environemtn.
    broker
        Instance of tradingenv.broker.Broker storing current holdings,
        net liquidation value, pnl, past track record, past rebalancing
        requests, past trades, past commissions and more.
    """
    # Type hinting here, set in Feature.reset
    exchange: 'tradingenv.exchange.Exchange' = None
    action_space: 'tradingenv.spaces.PortfolioSpace' = None
    broker: 'tradingenv.broker.Broker' = None

    def __init__(
            self,
            space: gymnasium.spaces.Space = None,
            name: str = None,
            save: bool = True,
            transformer = None,
    ):
        """
        Parameters
        ----------
        space
            An optional Space needed for OpenAI-gym compatibility. Note:
            only the feature before transformations (if any) will be validated
            against the space.
        name
            An optional name of the feature, class name by default. It is
            useful to provide a custom name to differentiate the same feature
            in the state if provided more than once e.g. with different
            parameters.
        save
            If True (default), the output of Feature.parse() will be
            automatically saved to feature.history whenever (1) an observed
            event is processed by the feature or (2) feature() is called.
        transformer
            A sklearn preprocessing transformer, defaults to None. If provided,
            by default the features returned by parse will be transformed.
            The easiest way to fitting transformers is to pass
            fit_transformer=True when instancing TradingEnv. You can set
            custom routines either by passing extending transformers or by
            implementing Feature._manual_fit_transformer.
        """
        # Note: optionally provide support for transformer.fit_partial as a
        # 'streaming' fit. Beware of double counting depending on the approach
        # used to call .fit_partial.
        if space is None and transformer is not None:
            raise ValueError(
                "You must provide a 'space' in order to use 'transformer'."
            )
        self.space = space
        self.name = name or self.name
        self.transformer = transformer
        self.save = save
        self.history: Dict[datetime, Any] = dict()

    def __call__(self, verify: bool = False, transform: bool = True):
        """
        Parameters
        ----------
        verify
            If False (default), the (un)transformed feature if validated
            against the space. The transformed feature is never validated
            against the transformed space as it's responsibility of the
            transformer to run further checks if any. In other words, it's
            possible for a transformed feature to fall outside the transformed
            space without raising errors.
            Here obs does not belong to space but calling feature() does not
            raise by default. This is by design for two reasons:
            (1) big features might not belong to the space until they warm up
            (2) state is verify=True by default, so avoid double verification.
        transform
            If True (default) and if a transformer is provided during
            initialisation of this class, then the feature will be transformed.

        Returns
        -------
        Parsed feature compatible with Feature.space, either transformed or not.
        If Feature.parse has not been implemented, returns self.
        """
        # TODO: you are not using 'transform' input.
        # TODO: when warming up batch transformer, transform must be = False
        # TODO: if transformation occurs, what do we save to history? Test.
        try:
            feature = self.parse()
        except NotImplementedError:
            feature = self
        else:
            verify = verify and self.space is not None
            if verify and (feature not in self.space):
                raise ValueError(
                    'The following state does not belong to the observation '
                    'space.\n{}'.format(repr(feature))
                )
            try:
                # 1.79 µs ± 10.5 ns per loop
                feature = self.transformer.transform(feature)
            except (NotFittedError, AttributeError):
                # AttributeError: 'NoneType' object has no attribute 'transform'
                # Copy so that to avoid changing history if parse returns a
                # mutable object (e.g. dict) that changes with new events.
                feature = copy.deepcopy(feature)
            finally:
                self._save_observation(feature)
        return feature

    def _now(self):
        """If the feature is not observing events but parsing some data from
        broker or exchange, feature.last_update will be none will be none
        while time is progressing. That is the reason why we seek if time
        has progressed in other parts of the environment. This is also a strong
        red flag that we would need a centralised Clock in the simulator."""
        # TODO: test.
        now = self.last_update
        if self.exchange is not None:
            if self.exchange.last_update is not None:
                now = max(now or datetime.min, self.exchange.last_update)
        return now

    def reset(
            self,
            exchange: 'tradingenv.exchange.Exchange' = None,
            action_space: 'tradingenv.spaces.PortfolioSpace' = None,
            broker: 'tradingenv.broker.Broker' = None,
    ):
        """Reset the feature to its original state. We pass exchange,
        action_space and broker as they could be needed to compute some features
        such as current positions, portfolio weights, market prices etc."""
        # TODO: Test.
        _transfomer = self.transformer
        super().reset()
        # Make the transformer persistent across resets.
        self.transformer = _transfomer
        self.exchange = exchange
        self.action_space = action_space
        self.broker = broker

    def _save_observation(self, obs):
        # TODO: history could be extended to class History(dict) implementing
        # extra methods to parse and visualise.
        if self.save:
            self.history[self._now()] = obs

    def _parse_history(self, make_2d: bool = False):
        """Parse history of the feature in a format that can be used to
        fit the transformer."""
        if issubclass(type(self.space), gymnasium.spaces.Box):
            # If space shape is (x, y, z, ...), shape of data will
            # be (n, x, y, z, ...).
            if len(self.history) == 0:
                raise ValueError(
                    f"You are trying to fit the transformer of feature "
                    f"{self.name} on a empty history. The easiest way to "
                    f"fix this is to pass fit_transformers=True when "
                    f"instancing TradingEnv."
                )
            history = list(self.history.values())
            data = np.concatenate([history])
            if make_2d:
                # Transformers expect 2D data. Meaning that (m, n) shaped
                # observations should be flattened
                # ValueError: Expected 2D array, got XD array instead
                # This operation transforms features from (nr_obs, m, n) to
                # (nr_obs, m * n).
                n, rows, cols = data.shape
                # https://stackoverflow.com/questions/53870113
                # ValueError: Found array with dim 3. StandardScaler expected <= 2.
                # So we squeeze
                size_flattened_feature = rows * cols
                data = data.reshape(n, size_flattened_feature)
        elif issubclass(type(self.space), gymnasium.spaces.MultiBinary):
            # No need to parse history as there is no need to fit a transformer
            # on dummy variables.
            data = None
        else:
            raise NotImplementedError(
                "You are attempting to fit a transformer in an unsupported "
                "space: {self.space}. A space must be supported to "
                "retrieve batch observations to fit. Use a support space "
                "of implement Feature._manual_fit_transformer."
                "Supported spaces are: {gymnasium.spaces.Box}."
            )
        return data

    def fit_transformer(self):
        """Fit transformed using all historical observations (batch) by default.
        The user can optionally implement _manual_fit_transformer to
        override the procedure."""
        try:
            self._manual_fit_transformer()
        except NotImplementedError:
            data = self._parse_history(make_2d=True)
            self.transformer.fit(data)

    def parse(self):
        """Returns any data structure representing the current value assumed
        by the feature. If a 'space' is provided when instancing this Feature,
        the returned value will be validated against the space if verify=True.
        This method is also required if you desire to store historical values
        of the feature in Feature.history whenever an observed event is
        received."""
        raise NotImplementedError()

    def _manual_fit_transformer(self):
        """You can implement here custom procedures to fit the transformer
        when Feature.fit_transformer is called."""
        raise NotImplementedError()
