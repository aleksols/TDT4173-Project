import requests
import pandas as pd
import io
import config
from tqdm import tqdm
from os import listdir, mkdir
import ta

def collect_data():
    """
    Get the daily, weekly and monthly time series data from Yahoo finance:
    https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
    :return: tuple with daily, weekly and monthly data
    """
    data = []

    for interval in tqdm(["1d", "1wk", "1mo"], desc="Downloading data"):
        params = {
            "period1": config.period_start,
            "period2": config.period_end,
            "interval": interval,
            "events": "history"
        }

        r = requests.get(
            url="https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC",
            params=params
        )

        f = io.BytesIO(r.content)  # Turn response into csv file
        try:
            d = pd.read_csv(f, parse_dates=[0], index_col=0)  # Turn csv file into dataframe

            # Remove unused columns
            d = d.drop("Adj Close", axis=1) \
                .drop("Open", axis=1) \
                .drop("High", axis=1) \
                .drop("Low", axis=1) \
                .drop("Volume", axis=1)

            data.append(d)
        except IOError:
            print("IO error")
            exit(-1)
    return data[0], data[1], data[2]


def add_indicators(data: pd.DataFrame):
    """
    Add the indicators RSI and MACD and return a new DataFrame object
    :param data: dataframe containing the time series data
    :return: dataframe with technical indicators added
    """
    macd = ta.trend.MACD(data.Close)
    rsi = ta.momentum.RSIIndicator(data.Close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_histogram"] = macd.macd_diff()
    data["rsi"] = rsi.rsi()
    return ta.utils.dropna(data)


def save_data(data: pd.DataFrame, file_name: str):
    """
    Save the dataframe as a pickled file
    :param data: pandas dataframe to be stored as a csv
    :param file_name: name for file to be saved
    :return: None
    """

    if config.data_folder not in listdir("."):
        mkdir(config.data_folder)
    data.to_pickle(f"{config.data_folder}/{file_name}")


if __name__ == '__main__':
    daily, weekly, monthly = collect_data()
    daily = add_indicators(daily)
    weekly = add_indicators(weekly)
    monthly = add_indicators(monthly)
    save_data(daily, "daily_data.pkl")
    save_data(weekly, "weekly_data.pkl")
    save_data(monthly, "monthly_data.pkl")
