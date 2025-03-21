# Release Notes


<!---
## [tradingenv x.y.z] - 2023-09-13

First release!

### Added

- `transformer` parameter of `tradingenv.env.TradingEnvXY` can now be set to 
`None` in order to prevent any transformation of the input X data.

### Changed

- In `tradingenv.env.TradingEnvXY`, data `X` is re-indexed to include 
timesteps from `Y`. This is done in order to avoid errors on missing data when 
there are gaps in `X`. It's users' responsibility to ensure that `X` does not 
have excessively gaps in the data.

### Removed

- TODO

### Fixed

- Fixed bug in `tradingenv.env.TradingEnvXY` that was causing an exception 
to be raised due to `start` not being dynamically inferred from X and Y data 
correctly when `window` was set to a value greater than 1.
- Rewards scaling was failing for out-of-sample data.
-->


## [tradingenv 0.1.3] - 2025-mm-dd

### Added

### Changed

- Discontinued backend `gym` in favor of `gymnasium`. This upgrade was long overdue as it was causing installation issues in some circumstances.

### Removed


### Fixed



## [tradingenv 0.1.2] - 2023-10-03


### Added

- Added `pandas.DataFrame.returns_by_month` to compute monthly returns.
- Added optional arguments risk_version and reward_clipping to 
  `tradingenv.env.TradingEnvXY`.


### Fixed

- README file was not being shown correctly in the PyPI page due to an error 
in pyproject.toml.


## [tradingenv 0.1.1] - 2023-10-02


### Added

- Added class `TradingEnvXY`, a high-level implementation of `TradingEnv` that 
  allows to pass features in tabular form (X) and prices for assets to be 
  traded (Y) as pandas.DataFrame.
- Added class `EventNewObservation` to `Transmitter` to notify the environment
  that a new observation is available. This makes it easier to create custom 
  environments whose observations can be expressed in tabular form.
- Reward class `LogReturn` now supports extra arguments to facilitate the 
  scaling and clipping of the reward within a predefined range.
- Added class `State`, an implementation of `IState` which makes it easier to
  get started with use cases where observations can be expressed in tabular 
  form.
- Added attribute `TradingEnv.visits` to keep track of the number of visits to
  each state in the environment during training.
- Added optional argument `episode_length` to `TradingEnv` to specify the 
  maximum number of steps in an episode. This is useful for environments where
  the default episode length is too large.
- Added docstrings and documentation.

### Changed

- Refactored contracts in `tradingenv.contract` module.


## [tradingenv 0.1.0] - 2023-09-13

First release!
<!---
### Added

- TODO

### Changed

- TODO

### Removed

- TODO

### Fixed

- TODO
-->