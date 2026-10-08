from backtesting import Strategy, Backtest
import yfinance as yf

from strategy import sma_crossover_signal, sma_derivative_signal

yf.Ticker("AAPL").history(period="1y")

class SmaCrossover(Strategy):
  
  n1 = 9
  n2 = 20
  
  def init(self):
    self.signals = self.I(
            sma_crossover_signal, self.data.Close, self.n1, self.n2
        )

  def next(self):
    if self.signals == 1:
      self.position.close()
      self.buy()
    elif self.signals == -1:
      self.position.close()
      self.sell()

class SmaDerivative(Strategy):
    def init(self):
        self.signals = self.I(sma_derivative_signal, self.data.Close, self.data.Volume, rolling_window_cross20=5, min_volume=1000000, min_price=10)

    def next(self):
        if self.signals == 1:
            self.buy()
        elif self.signals == -1:
            self.position.close()

data = yf.Ticker("AAPL").history(period="1y")
bt = Backtest(data, SmaDerivative, cash=1000, commission=.002)
stats = bt.run()
print(stats)
bt.plot()