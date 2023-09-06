"""Provides an extented python dictionary with handy methods when the dict
is mapping contracts to quantities (either nr of contracts of portfolio weight).
"""
import tradingenv
from tradingenv.contracts import AbstractContract, Cash
from typing import Sequence, Dict, Union
from numbers import Number
from abc import abstractmethod


class _Allocation(dict):
    """Clients should instance subclasses of this class. This is a general
    purpose class which is agnostic with respect to the unit measure of the
    allocation, so they could be weights, nr of contracts etc."""

    def __init__(
            self,
            mapping: Dict[AbstractContract, float] = None,
            keys: Sequence[AbstractContract] = None,
            values: Sequence[float] = None,
    ):
        """To initialize allocation you need to pass either 'mapping' OR
        'keys' and 'values'.

        Parameters
        ----------
        mapping
            A dictionary mapping instances of AbstractContract to a float
            representing the target allocation expressed in a whatever unit of
            measurement (e.g. weight or number of contracts).
        keys
            A sequence of AbstractContract instances, representing the keys of
            the dictionary to be constructed.
        values
            A sequence of floats, representing the values of the dictionary
            to be constructed.
        """
        # Define the generator.
        if mapping is not None:
            generator = mapping.items()
        elif keys is not None and values is not None:
            if len(keys) != len(values):
                raise ValueError("Keys and values must have the same length.")
            generator = zip(keys, values)
        else:
            generator = dict().items()

        # Use the generator.
        data = {
            contract.static_hashing(): value
            for contract, value in generator
            if not isinstance(contract, Cash)
            if value != 0
        }
        super().__init__(data)

    def __sub__(self, other: Union[Number, '_Allocation']):
        """Roughly equivalent to pandas.Series.__sub__. This allows to perform
        subtractions between dictionaries."""
        # I was going to implement several operators, but then realized that
        # I was re-inventing the wheel so I stopped. Not discontinuing
        # because why not.
        cls = type(self)
        if isinstance(other, Number):
            mapping = {k: v - other for k, v in self.items()}
        elif isinstance(other, cls):
            mapping = self.copy()
            for k, v in other.items():
                mapping[k] = mapping.get(k, 0) - v
        else:
            raise TypeError('Unsupported type {}'.format(other))
        return cls(mapping)

    @abstractmethod
    def _to_weights(self, broker: "tradingenv.broker.Broker") -> 'Weights':
        """Convert this instance to an instance of type Weights."""
        raise NotImplementedError()

    @abstractmethod
    def _to_nr_contracts(self, broker: "tradingenv.broker.Broker") -> 'NrContracts':
        """Convert this instance to an instance of type Weights."""
        raise NotImplementedError()


class Weights(_Allocation):
    """A dictionary mapping contracts to portfolio weight."""

    def _to_weights(self, broker: "tradingenv.broker.Broker") -> 'Weights':
        """Weights converted to weights returns a copy of self."""
        return Weights(self)

    def _to_nr_contracts(self, broker: "tradingenv.broker.Broker") -> 'NrContracts':
        """Map the current dictionary of portfolio weights to a dictionary of
        number of contracts."""
        nr_contracts = dict()
        nlv = broker.net_liquidation_value()
        for contract, weight in self.items():
            avg_price = broker.exchange[contract].acq_price(weight)
            nr_contracts[contract] = weight * nlv / avg_price / contract.multiplier
        return NrContracts(nr_contracts)


class NrContracts(_Allocation):
    """A dictionary mapping contracts to number of contracts."""

    def _to_weights(self, broker: "tradingenv.broker.Broker") -> 'Weights':
        """Map the current dictionary of number of contracts to a dictionary of
        portfolio weights."""
        weights = dict()
        nlv = broker.net_liquidation_value()
        for contract, quantity in self.items():
            order_book = broker.exchange[contract]
            prices = order_book.acq_price(quantity)
            weights[contract] = contract.multiplier * quantity * prices / nlv
        return Weights(weights)

    def _to_nr_contracts(self, broker: "tradingenv.broker.Broker") -> 'NrContracts':
        """NrContracts converted to number of contracts returns a copy of
        self."""
        return NrContracts(self)
