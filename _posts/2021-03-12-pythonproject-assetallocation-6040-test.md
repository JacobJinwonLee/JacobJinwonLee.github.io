자산배분은 상관관계가 낮거나 음수인 우상향하는 자산들에 분산 투자하는 것입니다. 가장 기본적이고, 잘 알려진 방식은 주식 60 : 채권 40 방식입니다.


```python
import pandas as pd
import pandas_datareader.data as web
import datetime
import backtrader as bt
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import pyfolio as pf
import quantstats
import math
plt.rcParams["figure.figsize"] = (10, 6) # (w, h)
```

    C:\ProgramData\Anaconda3\lib\site-packages\pyfolio\pos.py:27: UserWarning: Module "zipline.assets" not found; mutltipliers will not be applied to position notionals.
      'Module "zipline.assets" not found; mutltipliers will not be applied' +
    

ETF로 매수하는 것이 간편하니 적절한 ETF 데이터를 다운받습니다. 미국 전체 주식 ETF인 VTI와 미국 7-10년 만기 국채 IEF를 사용합니다.


```python
start = '2002-08-02'
end = '2021-03-11'
vti = web.DataReader("VTI", 'yahoo', start, end)['Adj Close'].to_frame("vti_Close")
ief = web.DataReader("IEF", 'yahoo', start, end)['Adj Close'].to_frame("ief_Close")
```


```python
vti.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vti_Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2002-08-01</th>
      <td>29.226831</td>
    </tr>
    <tr>
      <th>2002-08-02</th>
      <td>28.546240</td>
    </tr>
    <tr>
      <th>2002-08-05</th>
      <td>27.594112</td>
    </tr>
    <tr>
      <th>2002-08-06</th>
      <td>28.493328</td>
    </tr>
    <tr>
      <th>2002-08-07</th>
      <td>28.941196</td>
    </tr>
  </tbody>
</table>
</div>




```python
ief.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ief_Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2002-08-01</th>
      <td>47.479900</td>
    </tr>
    <tr>
      <th>2002-08-02</th>
      <td>47.846680</td>
    </tr>
    <tr>
      <th>2002-08-05</th>
      <td>48.087341</td>
    </tr>
    <tr>
      <th>2002-08-06</th>
      <td>47.697678</td>
    </tr>
    <tr>
      <th>2002-08-07</th>
      <td>47.846680</td>
    </tr>
  </tbody>
</table>
</div>



일단 모델 포트폴리오로, 매일 60:40 비중을 맞추는 것으로 생각하고 만듭니다. 거래비용은 생략합니다.


```python
vti_return = vti.pct_change(periods=1)
ief_return = ief.pct_change(periods=1)
df_return = pd.concat([vti_return, ief_return], axis=1)

df_return.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vti_Close</th>
      <th>ief_Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2002-08-01</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2002-08-02</th>
      <td>-0.023287</td>
      <td>0.007725</td>
    </tr>
    <tr>
      <th>2002-08-05</th>
      <td>-0.033354</td>
      <td>0.005030</td>
    </tr>
    <tr>
      <th>2002-08-06</th>
      <td>0.032587</td>
      <td>-0.008103</td>
    </tr>
    <tr>
      <th>2002-08-07</th>
      <td>0.015718</td>
      <td>0.003124</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_return['6040_return'] = df_return['vti_Close']*0.6 + df_return['ief_Close']*0.4
df_return.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vti_Close</th>
      <th>ief_Close</th>
      <th>6040_return</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2002-08-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2002-08-02</th>
      <td>-0.023287</td>
      <td>0.007725</td>
      <td>-0.010882</td>
    </tr>
    <tr>
      <th>2002-08-05</th>
      <td>-0.033354</td>
      <td>0.005030</td>
      <td>-0.018000</td>
    </tr>
    <tr>
      <th>2002-08-06</th>
      <td>0.032587</td>
      <td>-0.008103</td>
      <td>0.016311</td>
    </tr>
    <tr>
      <th>2002-08-07</th>
      <td>0.015718</td>
      <td>0.003124</td>
      <td>0.010681</td>
    </tr>
  </tbody>
</table>
</div>




```python
quantstats.reports.plots(df_return['6040_return'], mode='basic')
```


![output_9_0](https://user-images.githubusercontent.com/54884755/110947847-0358b900-8384-11eb-80b4-87d94aa90f8d.png)



![output_9_1](https://user-images.githubusercontent.com/54884755/110947937-19ff1000-8384-11eb-8ed5-6b5f2034e5c6.png)


매일 60:40 비중을 맞춘 결과 연 복리 수익률 9.23%, 샤프 비율 0.87, MDD -31% 정도입니다.


```python
quantstats.reports.metrics(df_return['6040_return'], mode='full')
```

                               Strategy
    -------------------------  ----------
    Start Period               2002-08-01
    End Period                 2021-03-11
    Risk-Free Rate             0.0%
    Time in Market             100.0%
    
    Cumulative Return          417.94%
    CAGR%                      9.23%
    Sharpe                     0.87
    Sortino                    1.24
    Max Drawdown               -31.65%
    Longest DD Days            895
    Volatility (ann.)          10.86%
    Calmar                     0.29
    Skew                       -0.1
    Kurtosis                   12.97
    
    Expected Daily %           0.04%
    Expected Monthly %         0.74%
    Expected Yearly %          8.57%
    Kelly Criterion            8.66%
    Risk of Ruin               0.0%
    Daily Value-at-Risk        -1.09%
    Expected Shortfall (cVaR)  -1.09%
    
    Payoff Ratio               0.92
    Profit Factor              1.18
    Common Sense Ratio         1.13
    CPC Index                  0.61
    Tail Ratio                 0.95
    Outlier Win Ratio          4.2
    Outlier Loss Ratio         4.08
    
    MTD                        1.73%
    3M                         3.17%
    6M                         11.38%
    YTD                        2.04%
    1Y                         25.35%
    3Y (ann.)                  12.27%
    5Y (ann.)                  12.06%
    10Y (ann.)                 10.52%
    All-time (ann.)            9.23%
    
    Best Day                   7.42%
    Worst Day                  -5.81%
    Best Month                 8.02%
    Worst Month                -10.48%
    Best Year                  21.47%
    Worst Year                 -16.97%
    
    Avg. Drawdown              -1.11%
    Avg. Drawdown Days         17
    Recovery Factor            13.21
    Ulcer Index                inf
    
    Avg. Up Month              1.93%
    Avg. Down Month            -2.02%
    Win Days %                 56.19%
    Win Month %                70.54%
    Win Quarter %              76.0%
    Win Year %                 90.0%
    

위에서 한 것처럼 그냥 만들어도 되지만, 백테스트에 많이 쓰이는 Backtrader 패키지를 한번 사용해 보겠습니다. Input 형식을 맞추어야 합니다.


```python
vti = vti.rename({'vti_Close':'Close'}, axis='columns')
ief = ief.rename({'ief_Close':'Close'}, axis='columns')

for column in ['Open', 'High', "Low"]:
    vti[column] = vti["Close"]
    ief[column] = ief["Close"]
```


```python
vti.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2002-08-01</th>
      <td>29.226831</td>
      <td>29.226831</td>
      <td>29.226831</td>
      <td>29.226831</td>
    </tr>
    <tr>
      <th>2002-08-02</th>
      <td>28.546240</td>
      <td>28.546240</td>
      <td>28.546240</td>
      <td>28.546240</td>
    </tr>
    <tr>
      <th>2002-08-05</th>
      <td>27.594112</td>
      <td>27.594112</td>
      <td>27.594112</td>
      <td>27.594112</td>
    </tr>
    <tr>
      <th>2002-08-06</th>
      <td>28.493328</td>
      <td>28.493328</td>
      <td>28.493328</td>
      <td>28.493328</td>
    </tr>
    <tr>
      <th>2002-08-07</th>
      <td>28.941196</td>
      <td>28.941196</td>
      <td>28.941196</td>
      <td>28.941196</td>
    </tr>
  </tbody>
</table>
</div>



60 : 40 비율로 매수하고 20 거래일마다 리밸런싱하는 전략입니다. 


```python
class AssetAllocation_6040(bt.Strategy):
    params = (
        ('equity',0.6),
    )
    def __init__(self):
        self.VTI = self.datas[0]
        self.IEF = self.datas[1]
        self.counter = 0
        
    def next(self):
        if  self.counter % 20 == 0:
            self.order_target_percent(self.VTI, target=self.params.equity)
            self.order_target_percent(self.IEF, target=(1 - self.params.equity))
        self.counter += 1
```


```python
cerebro = bt.Cerebro()

cerebro.broker.setcash(1000000)

VTI = bt.feeds.PandasData(dataname = vti)
IEF = bt.feeds.PandasData(dataname = ief)

cerebro.adddata(VTI)
cerebro.adddata(IEF)

cerebro.addstrategy(AssetAllocation_6040)

cerebro.addanalyzer(bt.analyzers.PyFolio, _name = 'PyFolio')

results = cerebro.run()
strat = results[0]

portfolio_stats = strat.analyzers.getbyname('PyFolio')
returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
returns.index = returns.index.tz_convert(None)

#quantstats.reports.html(returns, output = 'Report_AssetAllocation_6040.html', title='AssetAllocation_6040')
```


```python
quantstats.reports.plots(returns, mode='basic')
```


![output_18_0](https://user-images.githubusercontent.com/54884755/110948027-3602b180-8384-11eb-8ea4-5120b93b43a5.png)



![output_18_1](https://user-images.githubusercontent.com/54884755/110948083-4450cd80-8384-11eb-8caa-e28f013a6218.png)


20 거래일마다 리밸런싱하는 것으로 바꾸니 연 복리 수익률 9.01%, 샤프 비율 0.94, MDD -26%로 나옵니다.


```python
quantstats.reports.metrics(returns, mode='full')
```

                               Strategy
    -------------------------  ----------
    Start Period               2002-08-01
    End Period                 2021-03-11
    Risk-Free Rate             0.0%
    Time in Market             100.0%
    
    Cumulative Return          398.6%
    CAGR%                      9.01%
    Sharpe                     0.94
    Sortino                    1.33
    Max Drawdown               -26.04%
    Longest DD Days            917
    Volatility (ann.)          9.68%
    Calmar                     0.35
    Skew                       -0.37
    Kurtosis                   8.03
    
    Expected Daily %           0.03%
    Expected Monthly %         0.72%
    Expected Yearly %          8.36%
    Kelly Criterion            8.85%
    Risk of Ruin               0.0%
    Daily Value-at-Risk        -0.97%
    Expected Shortfall (cVaR)  -0.97%
    
    Payoff Ratio               0.93
    Profit Factor              1.19
    Common Sense Ratio         1.16
    CPC Index                  0.62
    Tail Ratio                 0.98
    Outlier Win Ratio          3.83
    Outlier Loss Ratio         3.8
    
    MTD                        1.78%
    3M                         3.32%
    6M                         11.58%
    YTD                        2.19%
    1Y                         23.99%
    3Y (ann.)                  11.7%
    5Y (ann.)                  11.69%
    10Y (ann.)                 10.18%
    All-time (ann.)            9.01%
    
    Best Day                   5.28%
    Worst Day                  -5.23%
    Best Month                 7.07%
    Worst Month                -9.18%
    Best Year                  20.71%
    Worst Year                 -14.9%
    
    Avg. Drawdown              -1.07%
    Avg. Drawdown Days         18
    Recovery Factor            15.31
    Ulcer Index                inf
    
    Avg. Up Month              1.87%
    Avg. Down Month            -1.88%
    Win Days %                 55.99%
    Win Month %                70.09%
    Win Quarter %              76.0%
    Win Year %                 90.0%
    

월간 데이터를 사용하면 훨씬 더 과거의 결과도 테스트해 볼 수 있습니다. 가장 긴 시계열의 경우 1900년 1월부터 2020년 12월까지의 데이터가 있습니다.


```python
MonthlyReturn = pd.read_excel('MonthlyAssetClassReturn.xlsx')
```


```python
MonthlyReturn.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Data Index</th>
      <th>Broker Call Rate</th>
      <th>CPI</th>
      <th>T-Bills</th>
      <th>S&amp;P 500 Total return</th>
      <th>Small Cap Stocks</th>
      <th>MSCI EAFE</th>
      <th>EEM</th>
      <th>US 10 YR</th>
      <th>US Corp Bond Return Index</th>
      <th>...</th>
      <th>International Small Cap Value (Global B/M Small Low)</th>
      <th>International Large Cap Value (Global B/M Big Low)</th>
      <th>International Small High Mom (Global mom Small High)</th>
      <th>International Large High Mom (Global mom Small High)</th>
      <th>Merrill High Yield</th>
      <th>World Stocks</th>
      <th>World ex USA</th>
      <th>BuyWrite</th>
      <th>PutWrite</th>
      <th>Bitcoin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1900-01-31</td>
      <td>NaN</td>
      <td>0.013333</td>
      <td>0.0025</td>
      <td>0.016413</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1900-02-28</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0025</td>
      <td>0.021138</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.011278</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1900-03-31</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0025</td>
      <td>0.011084</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.009758</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1900-04-30</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0025</td>
      <td>0.015894</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.016107</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1900-05-31</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0025</td>
      <td>-0.044246</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.016023</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



시계열로 바꾸어 주는 것이 사용하기 편합니다. 1열인 Data Index가 월말 날짜이므로, 이 열을 인덱스로 잡습니다.


```python
MonthlyReturn = MonthlyReturn.set_index('Data Index')
```


```python
MonthlyReturn.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Broker Call Rate</th>
      <th>CPI</th>
      <th>T-Bills</th>
      <th>S&amp;P 500 Total return</th>
      <th>Small Cap Stocks</th>
      <th>MSCI EAFE</th>
      <th>EEM</th>
      <th>US 10 YR</th>
      <th>US Corp Bond Return Index</th>
      <th>GSCI</th>
      <th>...</th>
      <th>International Small Cap Value (Global B/M Small Low)</th>
      <th>International Large Cap Value (Global B/M Big Low)</th>
      <th>International Small High Mom (Global mom Small High)</th>
      <th>International Large High Mom (Global mom Small High)</th>
      <th>Merrill High Yield</th>
      <th>World Stocks</th>
      <th>World ex USA</th>
      <th>BuyWrite</th>
      <th>PutWrite</th>
      <th>Bitcoin</th>
    </tr>
    <tr>
      <th>Data Index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1900-01-31</th>
      <td>NaN</td>
      <td>0.013333</td>
      <td>0.0025</td>
      <td>0.016413</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1900-02-28</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0025</td>
      <td>0.021138</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.011278</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1900-03-31</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0025</td>
      <td>0.011084</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.009758</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1900-04-30</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0025</td>
      <td>0.015894</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.016107</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1900-05-31</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0025</td>
      <td>-0.044246</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.016023</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



필요한 것만 뽑아옵니다. 월간 미국 주식(S&P 500), 월간 미국 10년 만기 국채 수익률 데이터입니다. 1900년 1월부터 2020년 12월까지 121년 기간의 테스트가 될 것입니다.


```python
Monthly_6040 = MonthlyReturn.loc[:, ['S&P 500 Total return', 'US 10 YR']]
Monthly_6040.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S&amp;P 500 Total return</th>
      <th>US 10 YR</th>
    </tr>
    <tr>
      <th>Data Index</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1900-01-31</th>
      <td>0.016413</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1900-02-28</th>
      <td>0.021138</td>
      <td>0.011278</td>
    </tr>
    <tr>
      <th>1900-03-31</th>
      <td>0.011084</td>
      <td>0.009758</td>
    </tr>
    <tr>
      <th>1900-04-30</th>
      <td>0.015894</td>
      <td>-0.016107</td>
    </tr>
    <tr>
      <th>1900-05-31</th>
      <td>-0.044246</td>
      <td>0.016023</td>
    </tr>
  </tbody>
</table>
</div>




```python
Monthly_6040['Monthly_6040'] = Monthly_6040['S&P 500 Total return'] * 0.6 + Monthly_6040['US 10 YR'] * 0.4
Monthly_6040.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S&amp;P 500 Total return</th>
      <th>US 10 YR</th>
      <th>Monthly_6040</th>
    </tr>
    <tr>
      <th>Data Index</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1900-01-31</th>
      <td>0.016413</td>
      <td>0.000000</td>
      <td>0.009848</td>
    </tr>
    <tr>
      <th>1900-02-28</th>
      <td>0.021138</td>
      <td>0.011278</td>
      <td>0.017194</td>
    </tr>
    <tr>
      <th>1900-03-31</th>
      <td>0.011084</td>
      <td>0.009758</td>
      <td>0.010554</td>
    </tr>
    <tr>
      <th>1900-04-30</th>
      <td>0.015894</td>
      <td>-0.016107</td>
      <td>0.003094</td>
    </tr>
    <tr>
      <th>1900-05-31</th>
      <td>-0.044246</td>
      <td>0.016023</td>
      <td>-0.020139</td>
    </tr>
  </tbody>
</table>
</div>



월간 데이터이므로, 일간 데이터 기준인 패키지가 주는 값을 적절히 조정해야 합니다. 1년 12개월 252거래일을 가정합니다. 1900년 1월부터 121년 동안 샤프 비율은 0.766으로 나옵니다. 아래 그림의 제목 하단에 있는 샤프 비율은 무시하고, 직접 계산한 값을 보아야 합니다.


```python
quantstats.stats.sharpe(Monthly_6040['Monthly_6040'])/math.sqrt(252/12)
```




    0.7661757370916087




```python
quantstats.reports.plots(Monthly_6040['Monthly_6040'], mode='basic')
```


![output_32_0](https://user-images.githubusercontent.com/54884755/110948214-69ddd700-8384-11eb-848a-ec162f3e4dcb.png)



![output_32_1](https://user-images.githubusercontent.com/54884755/110948253-76fac600-8384-11eb-81b0-06b5c2b47ded.png)


연 복리 수익률 8.13%, 샤프 비율은 위에서 계산한대로 0.766, MDD는 대공황 시기 덕분에 -63%입니다. 매년 8%를 약간 넘는 수익이라도 121년을 하면 원금이 12800배가 됩니다. 60:40 전략은 대공황 시기가 조금 문제이지만 장기간 괜찮은 수익을 내면서도 간단한 좋은 전략입니다. 


```python
quantstats.reports.metrics(Monthly_6040['Monthly_6040'], mode='full')
```

                               Strategy
    -------------------------  -------------
    Start Period               1900-01-31
    End Period                 2020-12-31
    Risk-Free Rate             0.0%
    Time in Market             100.0%
    
    Cumulative Return          1,280,088.84%
    CAGR%                      8.13%
    Sharpe                     3.51
    Sortino                    5.77
    Max Drawdown               -63.53%
    Longest DD Days            2496
    Volatility (ann.)          50.5%
    Calmar                     0.13
    Skew                       0.29
    Kurtosis                   8.85
    
    Expected Daily %           0.65%
    Expected Monthly %         0.65%
    Expected Yearly %          8.13%
    Kelly Criterion            29.1%
    Risk of Ruin               0.0%
    Daily Value-at-Risk        -4.53%
    Expected Shortfall (cVaR)  -4.53%
    
    Payoff Ratio               1.09
    Profit Factor              1.86
    Common Sense Ratio         2.26
    CPC Index                  1.28
    Tail Ratio                 1.22
    Outlier Win Ratio          3.26
    Outlier Loss Ratio         3.27
    
    MTD                        2.0%
    3M                         4.13%
    6M                         12.11%
    YTD                        16.29%
    1Y                         16.29%
    3Y (ann.)                  12.02%
    5Y (ann.)                  11.01%
    10Y (ann.)                 10.6%
    All-time (ann.)            8.13%
    
    Best Day                   25.45%
    Worst Day                  -19.0%
    Best Month                 25.45%
    Worst Month                -19.0%
    Best Year                  32.88%
    Worst Year                 -31.17%
    
    Avg. Drawdown              -4.16%
    Avg. Drawdown Days         167
    Recovery Factor            20149.0
    Ulcer Index                0.99
    
    Avg. Up Month              2.42%
    Avg. Down Month            -2.22%
    Win Days %                 63.02%
    Win Month %                63.02%
    Win Quarter %              69.01%
    Win Year %                 77.69%
    
