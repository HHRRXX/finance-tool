import os
import streamlit as st
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import pyfolio
from copy import deepcopy
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, get_daily_return, get_baseline
import itertools
from finrl import config_tickers
import os
from finrl.config import INDICATORS
   
df_initial=np.arange(100000,5010000,10000,int)
st.title("100 NASDAQ Stocks finance tool")
st.markdown("The user just input your **Trade Start Date** and **Trade End Date**, the System will analyze and draw your finance report")
start_date = st.slider('Your trade start date', datetime.date(2019,1,1), datetime.date(2020,10,1))
end_date = st.slider('Your trade end date', datetime.date(2020,10,2), datetime.date(2022,10,1))
initial_asset = st.selectbox(
    'Please Select Your Initial Asset', df_initial)
TRADE_START_DATE=start_date.__str__()
TRADE_END_DATE=end_date.__str__()

df = YahooDownloader(start_date = TRADE_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.NAS_100_TICKER).fetch_data()
st.success('Data obtain finished', icon="✅")

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)
st.success('Data processed finished', icon="✅")
list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
stock_dimension = len(trade.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": initial_asset,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}
e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
trained_sac_path = "pages/trained_models/trained_sac_100_full/"
df_account_value, df_actions = DRLAgent.DRL_prediction_load_from_file(
    model_name="sac", 
    environment = e_trade_gym,
    cwd=trained_sac_path)
st.success('Result get!', icon="✅")

#df_account_value.to_csv("FinRL-master/backend/data/account_value.csv")
#df_actions.to_csv("FinRL-master/backend/data/actions.csv")

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
#perf_stats_all.to_csv("FinRL-master/backend/data/states_pre.csv")

baseline_ticker="^NDX"
value_col_name="account_value"
baseline_start = df_account_value.loc[0,'date']
baseline_end = df_account_value.loc[len(df_account_value)-1,'date']
df = deepcopy(df_account_value)
df["date"] = pd.to_datetime(df["date"])
test_returns = get_daily_return(df, value_col_name=value_col_name)
baseline_df = get_baseline(ticker=baseline_ticker, start=baseline_start, end=baseline_end)
baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
baseline_returns = get_daily_return(baseline_df, value_col_name="close")

#test_returns.to_csv("FinRL-master/backend/data/test_return.csv")
#baseline_returns.to_csv("FinRL-master/backend/data/baseline_return.csv")

f=pyfolio.create_full_tear_sheet(returns=test_returns, benchmark_rets=baseline_returns, set_context=False)
#plt.savefig("FinRL-master/backend/image/return_compare.png")

fig= plt.figure()
plt.plot(test_returns.index,test_returns.values)
plt.title("daily return")
plt.xlabel('date')
plt.xticks(rotation=90)
#plt.savefig("FinRL-master/backend/image/daily_return.png")

st.success('The system is generating your finance report!', icon="✅")

st.write("Here's our profit everyday:")
st.write(df_account_value)
st.write("Here's our action everyday to 100 Nasdaq stocks:")
st.write(df_actions)
st.write("Here's our report to 100 Nasdaq stocks:")
st.write(perf_stats_all)
st.write("Here's our daily return to 100 Nasdaq stocks:")
st.write(test_returns)

st.write("Here's our finance analysis report:")
st.write(f)
st.write(fig)
st.balloons()
st.caption("Created by Yang Wenkai, Huang Runxing and Chen Haoyang")