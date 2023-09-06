# TODO: add test coverage badge here.

Introduction
============
Backtest trading strategies or train reinforcement learning agents with
:code:`tradingenv`, an event-driven market simulator.

**Backtesting**. Thanks to the event-driven design, tradingenv is agnostic with
respect to the type and time-frequency of the events. This means that you can
run simulations either using irregularly sampled trade and quotes data, daily
closing prices, monthly economic data or alternative data. Financial instruments
supported include stocks, ETF and futures.

**Reinforcement learning**. The package is built upon the industry-standard gym_ and therefore can be used
in conjunction with popular reinforcement learning frameworks including rllib_ 
and stable-baselines3_.


Example - Backtesting
=====================
TODO: Backtest 60:40 and show tearsheet.

Example - Reinforcement Learning
================================
TODO: Trade SPX using transaction costs. Show OpenAI - API.


Installation
============
tradingenv supports Python 3.8 or newer versions. The following command line
will install the latest software version.

.. code-block:: console

    pip install tradingenv

Notebooks, software tests and building the documentation require extra
dependencies that can be installed with

.. code-block:: console

    pip install tradingenv[extra]

You can optionally run the software tests typing :code:`pytest` in the command
line, assuming that the folder :code:`\tests` is in the current working directory.


Relevant projects
=================
- btgym_: is an OpenAI Gym-compatible environment for
- backtrader_ backtesting/trading library, designed to provide gym-integrated framework for running reinforcement learning experiments in [close to] real world algorithmic trading environments.
- gym_: A toolkit for developing and comparing reinforcement learning algorithms.
- rllib_: open-source library for reinforcement learning.
- stable-baselines3_: is a set of reliable implementations of reinforcement learning algorithms in PyTorch.


Developer notes
===============
You are welcome to contribute by creating issues and pull requests.

To refresh and build the documentation:

.. code-block::

   pytest tests/notebooks
   sphinx-apidoc -f -o docs/source tradingenv
   cd docs
   make clean
   make html


.. Hyperlinks.
.. _btgym: https://github.com/Kismuz/btgym
.. _backtrader: https://github.com/backtrader/backtrader
.. _gym: https://github.com/openai/gym
.. _rllib: https://docs.ray.io/en/latest/rllib/
.. _stable-baselines3: https://github.com/hill-a/stable-baselines
