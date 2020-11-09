import requests
import pandas as pd
import io
import config
from tqdm import tqdm
from os import listdir, mkdir


def collect_data():
    data = []

    for interval in tqdm(["1d", "1wk", "1mo"], desc="Downloading data"):
        params = {
            "period1": config.period_start,
            "period2": config.period_end,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "false"
        }

        r = requests.get(
            url="https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC",
            params=params
        )

        f = io.BytesIO(r.content)  # Turn response into csv file
        try:
            d = pd.read_csv(f)
            data.append(d)
        except IOError:
            print("IO error")
            exit(-1)
    return data[0], data[1], data[2]


def add_indicators(data):
    """
    Add the indicators RSI and MACD to the given dataframe object
    :param data: pandas dataframe
    :return: None
    """
    # TODO implement


def save_data(data: pd.DataFrame, file_name):
    """
    Save the dataframe as a csv file
    :param data: pandas dataframe to be stored as a csv
    :param file_name: name for file to be saved
    :return: None
    """

    if config.data_folder not in listdir("."):
        mkdir(config.data_folder)
    data.to_csv(f"{config.data_folder}/{file_name}")


if __name__ == '__main__':
    daily, weekly, monthly = collect_data()
    add_indicators(daily)
    add_indicators(weekly)
    add_indicators(monthly)
    save_data(daily, "daily_data.csv")
    save_data(weekly, "weekly_data.csv")
    save_data(monthly, "monthly_data.csv")
