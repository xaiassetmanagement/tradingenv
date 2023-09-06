# Futures vs ETFs backtesting from an implementation's perspective

#### 1. Futures expiration 
**Description:** In any point in time, we have to make sure that any 
future contract held in the account can be liquidated. A naive implementation 
that uses the expiry date to roll-over the contract is wrong, as in the real 
world brokers tend to stop support for trading as soon as the delivery process 
of the underlying starts (cutoff date). The cutoff varies with the broker and 
with the future contract. In the case of treasury futures, two business days 
prior to the first day allowed for deliveries into an expiring futures contract 
(ie first day of delivery month), clearing firms report to CME Clearing all open
long positions, grouped by account origin (customer or house) and position 
vintage date ([see reference](https://www.cmegroup.com/education/files/understanding-treasury-futures.pdf)).
A long notice periods tend to occur for all future contracts whose underlying 
is deliverable (e.g. see start of close-out period for Interactive Brokers 
[here](https://www.interactivebrokers.com/en/index.php?f=deliveryExerciseActions&p=physical)). 
For example, ZNM19 expires in date 2019-06-19 but Interactive Brokers 
liquidates any position from date 2019-05-28.


**Solution (implemented)**: each `Future` contract must be initialized with the cutoff 
date. An `EventContractDiscontinued` associated with the cutoff date will be 
added to the `Transmitter` when instancing `TradingEnv`, resetting all 
attributes of `LimitOrderBook`. Any attempt to retrieve prices (e.g. to 
calculate the NLV) will return `np.nan` thus raising an exception when used 
by `Broker`.  

#### 2. Automatic-rolling
**Description:** What if a contract is held until the expiry date?
**Solution:** `Future` instances must specify an expiry date and a cutoff date. 
Trading is not allowed after the cutoff date. After the expiry date, the limit 
order-book associated with the discontinued contract will be stopped, so 
attempts by `Broker` to calculate the NLV of the account will raise an 
exception as market prices are not available. Therefore, it's responsability of 
the user to do the roll-over, or the environment will raise an error. 
Alternatively, the user can use the AbstractContract class `FutureChain` and 
tradingenv will associated target weights with the leading contract.

#### 3. Multiplier
**Description:** the notional value of a future contract in *n* times the 
underlying index, where *n* is a fixed integer (e.g. 50 for the e-mini S&P 500).
How should this be implemented?

**Solution:** target weights are applied to the NLV, so target weights must not 
be affected by the multiplier. What needs to change is the calculation of the 
value of each position in the portfolio, which needs to be multiplied by the 
multiplier. At the same time, the number of shares to be traded needs to be 
divided by the multiplier.

#### 4. Margin
**Description:** Futures margin is the amount of money that you must deposit 
and keep on hand with your broker when you open a futures position. It is not 
a down payment and you do not own the underlying commodity. Futures margin 
generally represents a smaller percentage of the notional value of the 
contract, typically 3-12% per futures contract as opposed to up to 50% of 
the face value of securities purchased on margin. If the funds in your 
account drop below the maintenance margin level, you will receive a mergin coll 
or your position will be liquidated automatically.

**Solution:**