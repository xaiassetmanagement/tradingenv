.. figure:: https://tradingenv.blob.core.windows.net/images/logo-background-cropped.png
   :align: center



.. raw:: html

    <br/>
    <a href="https://https://github.com/xaiassetmanagement/tradingenv/actions/workflows/build-docs.yml">
        <img src="https://github.com/xaiassetmanagement/tradingenv/actions/workflows/build-docs.yml/badge.svg" alt="No message"/></a>
    <a href="https://github.com/xaiassetmanagement/tradingenv/actions/workflows/software-tests.yml">
        <img src="https://github.com/xaiassetmanagement/tradingenv/actions/workflows/software-tests.yml/badge.svg" alt="No message"/></a>
    <a href="https://github.com/xaiassetmanagement/tradingenv/actions">
        <img src="https://raw.githubusercontent.com/xaiassetmanagement/tradingenv/coverage-badge/coverage.svg" alt="No message"/></a>
    <br/>
    <a href="https://www.python.org">
        <img src="https://img.shields.io/pypi/pyversions/shap" alt="No message"/></a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="No message"/></a>

Introduction
============
Backtest trading strategies or train reinforcement learning agents with
:code:`tradingenv`, an event-driven market simulator that implements the
OpenAI/gym protocol.


Installation
============
tradingenv supports Python 3.7 or newer versions. The following command line
will install the latest software version.

.. code-block:: console

   pip install tradingenv

Notebooks, software tests and building the documentation require extra
dependencies that can be installed with

.. code-block:: console

   pip install tradingenv[extra]


Example - Reinforcement Learning
================================
The package is built upon the industry-standard gym_ and therefore can be used
in conjunction with popular reinforcement learning frameworks including rllib_
and stable-baselines3_.

.. only:: html

   .. literalinclude:: ../../tests/examples/test_readme.py
      :language: python
      :pyobject: TestReadme.test_instancing_env
      :tab-width: 0
      :dedent: 8
      :lines: 2-

.. only:: not html

   .. code-block:: python

      # Your Python code here
      import tradingenv
      print('Hello')



Example - Backtesting
=====================
Thanks to the event-driven design, tradingenv is agnostic with
respect to the type and time-frequency of the events. This means that you can
run simulations either using irregularly sampled trade and quotes data, daily
closing prices, monthly economic data or alternative data. Financial instruments
supported include stocks, ETF and futures.

.. only:: html

   .. literalinclude:: ../../tests/examples/test_readme.py
      :language: python
      :pyobject: TestReadme.test_backtest_60_40
      :dedent: 8
      :lines: 3-

.. figure:: https://tradingenv.blob.core.windows.net/images/tearsheet.png
|
.. figure:: https://tradingenv.blob.core.windows.net/images/fig_net_liquidation_value.png


Relevant projects
=================
- btgym_: is an OpenAI Gym-compatible environment for
- backtrader_ backtesting/trading library, designed to provide gym-integrated framework for running reinforcement learning experiments in [close to] real world algorithmic trading environments.
- gym_: A toolkit for developing and comparing reinforcement learning algorithms.
- qlib_: Qlib provides a strong infrastructure to support quant research.
- rllib_: open-source library for reinforcement learning.
- stable-baselines3_: is a set of reliable implementations of reinforcement learning algorithms in PyTorch.


Developers
==========
You are welcome to contribute features, examples and documentation or issues.

You can run the software tests typing :code:`pytest` in the command line,
assuming that the folder :code:`\tests` is in the current working directory.

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
.. _qlib: https://github.com/microsoft/qlib
.. _rllib: https://docs.ray.io/en/latest/rllib/
.. _stable-baselines3: https://github.com/hill-a/stable-baselines
