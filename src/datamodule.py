import os
from glob import glob

import numpy as np
import pandas as pd
import time
from datetime import timedelta
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def process_approach_data(filepath, USE_COLS):
    df = pd.read_csv(filepath)
    df['time'] = df['time'].replace('午前', '8:00')
    df['Timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.drop('cattle_ids', axis=1).join(
        df['cattle_ids'].str.split('-', expand=True).stack().reset_index(level=1, drop=True).rename('Cattle_id')
    )
    df['Interaction'] = 'approach'
    df = df[USE_COLS]
    return df

def process_body_fluids_data(filepath, USE_COLS):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'id': 'Cattle_id'})
    df['time'] = df['time'].replace('午前', '8:00')
    df['Timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.loc[df['fuilds'] != 0]
    df['Interaction'] = 'body_fuilds'
    df = df[USE_COLS]
    return df

def process_mounting_data(filepath, USE_COLS):
    df = pd.read_csv(filepath)
    df['time'] = df['time'].replace('午前', '8:00')
    df = df.rename(columns={'from': 'Cattle_id'})
    df['Interaction'] = 'mounting'
    df['Timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df[USE_COLS]
    return df

def process_standing_data(filepath, USE_COLS):
    df = pd.read_csv(filepath)
    df['time'] = df['time'].replace('午前', '8:00')
    df = df.loc[df['consent'] == 1]
    df['Cattle_id'] = df['to']
    df['Interaction'] = 'standing'
    df['Timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df[USE_COLS]
    return df

def concatenate_datasets(list_of_dfs):
    df_heat = pd.concat(list_of_dfs, axis=0).reset_index(drop=True)
    df_heat['Heat'] = 1
    # Project data to the next two days with modifications
    df_heat_future = df_heat.copy()
    df_heat_future['Timestamp'] += timedelta(days=1)
    df_heat_future['Heat'] = 0
    df_heat_future['Interaction'] = 0
    df_heat_future_plus_one = df_heat_future.copy()
    df_heat_future_plus_one['Timestamp'] += timedelta(days=1)
    return pd.concat([df_heat, df_heat_future, df_heat_future_plus_one], axis=0).sort_values('Timestamp').reset_index(drop=True)

def split_data(df, ratio):
    split_point = int(len(df) * ratio)
    return df.iloc[:split_point], df.iloc[split_point:]

def get_datadf(FILES_PATTERNS, USE_COLS):
    # Load and process data
    df_approach = process_approach_data(FILES_PATTERNS['approach'], USE_COLS) if 'approach' in FILES_PATTERNS else pd.DataFrame()
    df_body_fluids = process_body_fluids_data(FILES_PATTERNS['body_fluids'], USE_COLS) if 'body_fluids' in FILES_PATTERNS else pd.DataFrame()
    df_mounting = process_mounting_data(FILES_PATTERNS['mounting'], USE_COLS) if 'mounting' in FILES_PATTERNS else pd.DataFrame()
    df_standing = process_standing_data(FILES_PATTERNS['mounting'], USE_COLS) if 'standing' in FILES_PATTERNS else pd.DataFrame()
    # Combine all dataframes
    df = concatenate_datasets([df_approach, df_body_fluids, df_mounting, df_standing])
    # Only reliability data
    df = df.loc[df.loc[:, 'Timestamp'] >= '2023-10-1']
    return df


def wrap_preprocess(BASE_DIR, FILES_PATTERNS, USE_COLS, DICT_DEVICE, SENSOR_DIR, MODALITY, SAMPLING_RATE, USE_CHACE):
    if os.path.exists(os.path.join(BASE_DIR, 'processed_df.csv')) and USE_CHACE:
        print('Loading processed data')
        datadf = pd.read_csv(os.path.join(BASE_DIR, 'processed_df.csv'))
        datadf['Timestamp'] = pd.to_datetime(datadf['Timestamp'])
        return datadf
    datadf = get_datadf(FILES_PATTERNS, USE_COLS)
    for i, row in datadf.iterrows():
        row = dict(row)
        cattle_id = int(row['Cattle_id'])
        device_id = DICT_DEVICE.get(cattle_id)
        if device_id is None:
            print('[warning] input data is invalid(device id)')
            datadf = datadf.drop(i)
            continue
        date = row['Timestamp'].strftime('%Y%m%d')
        p_csv = os.path.join(SENSOR_DIR, f'cattle{str(device_id).zfill(4)}_{date}_{MODALITY}.csv')
        if not os.path.exists(p_csv):
            print('[warning] input data is invalid(missing file)')
            datadf = datadf.drop(i)
            continue
        df_x = pd.read_csv(p_csv, header=None)
        df_x.columns = ['Timestamp', 'Cattle_id', 'x', 'y', 'z']
        df_x['Timestamp'] = pd.to_datetime(df_x['Timestamp'])
        min_timestamp = row['Timestamp'] - pd.Timedelta(hours=1, minutes =30)
        max_timestamp = row['Timestamp'] + pd.Timedelta(hours=1, minutes =30)
        mask = (df_x['Timestamp'] >= min_timestamp) & (df_x['Timestamp'] <= max_timestamp)
        df_x = df_x.loc[mask].drop('Cattle_id', axis=1)
        range_timestamp = pd.date_range(start=min_timestamp, end=max_timestamp, freq=f'{int(1000/SAMPLING_RATE)}ms')
        df_timestamp = pd.DataFrame(index=range_timestamp)
        df_timestamp[['x', 'y', 'z']] = 0
        df_x = df_x.set_index('Timestamp')
        df_x = df_x.groupby(df_x.index).mean()
        df_x = df_x.sort_index()
        df_x = df_x.add(df_timestamp, fill_value=np.nan)
        df_x = df_x.interpolate(method='time', limit_direction='both', limit=None)
        if df_x.isna().sum().sum():
            print('[warning] input data is invalid(missing value)')
            datadf = datadf.drop(i)
            continue
    datadf.to_csv(os.path.join(BASE_DIR, 'processed_df.csv'))
    return datadf


class HeatCattleDataset(Dataset):

    def __init__(self, df, mode, dict_device, modality, sensor_dir, sampling_rate):
        self.df = df
        self.mode = mode
        self.dict_device = dict_device
        self.modality = modality
        self.sensor_dir = sensor_dir
        self.sampling_rate = sampling_rate

    def __getitem__(self, index):
        row = dict(self.df.iloc[index])
        y = torch.tensor([row['Heat']], dtype=torch.float32)
        cattle_id = int(row['Cattle_id'])
        device_id = self.dict_device.get(cattle_id)
        date = row['Timestamp'].strftime('%Y%m%d')
        p_csv = os.path.join(self.sensor_dir, f'cattle{str(device_id).zfill(4)}_{date}_{self.modality}.csv')
        df_x = pd.read_csv(p_csv, header=None)
        df_x.columns = ['Timestamp', 'Cattle_id', 'x', 'y', 'z']
        df_x['Timestamp'] = pd.to_datetime(df_x['Timestamp'])
        # Randomly shift the timestamp
        if self.mode == 'train':
            random_seconds = np.random.randint(-1800, 1801)
            row['Timestamp'] = row['Timestamp'] + pd.to_timedelta(random_seconds, unit='s')
        # Extract the data
        min_timestamp = row['Timestamp'] - pd.Timedelta(hours=1)
        max_timestamp = row['Timestamp'] + pd.Timedelta(hours=1)
        mask = (df_x['Timestamp'] >= min_timestamp) & (df_x['Timestamp'] <= max_timestamp)
        df_x = df_x.loc[mask].drop('Cattle_id', axis=1)
        df_x = df_x.set_index('Timestamp')
        df_x = df_x.groupby(df_x.index).mean()
        df_x = df_x.sort_index()
        # Drop the outliers
        min_q = 0.01
        max_q = 0.99
        quantiles = df_x[['x', 'y', 'z']].quantile([min_q, max_q])
        df_x = df_x[
            (df_x['x'].between(quantiles.loc[min_q, 'x'], quantiles.loc[max_q, 'x'])) &
            (df_x['y'].between(quantiles.loc[min_q, 'y'], quantiles.loc[max_q, 'y'])) &
            (df_x['z'].between(quantiles.loc[min_q, 'z'], quantiles.loc[max_q, 'z']))
        ]
        # Interpolate the data
        range_timestamp = pd.date_range(start=min_timestamp, end=max_timestamp, freq=f'{int(1000/self.sampling_rate)}ms')
        df_timestamp = pd.DataFrame(index=range_timestamp)
        df_timestamp[['x', 'y', 'z']] = 0
        df_x = df_x.add(df_timestamp, fill_value=np.nan)
        df_x = df_x.interpolate(method='time', limit_direction='both', limit=None)
        torch_x = torch.tensor(df_x.loc[:, ['x', 'y', 'z']].values, dtype=torch.float32).T
        return torch_x, y

    def __len__(self):
        return len(self.df)


class HeatCattleDM(pl.LightningDataModule):
    def __init__(self, setup_kargs):
        super().__init__()
        self.batch_size = 1
        self.setup_kargs = setup_kargs

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        df = wrap_preprocess(
            self.setup_kargs['BASE_DIR'],
            self.setup_kargs['FILES_PATTERNS'],
            self.setup_kargs['USE_COLS'],
            self.setup_kargs['DICT_DEVICE'],
            self.setup_kargs['SENSOR_DIR'],
            self.setup_kargs['MODALITY'],
            self.setup_kargs['SAMPLING_RATE'],
            self.setup_kargs['USE_CHACE']
        )
        df_train, df_valid = split_data(df, self.setup_kargs['TRAIN_RATIO'])
        print(f'Training set size: {len(df_train)}, Validation set size: {len(df_valid)}')
        print()
        print(f'Number of positive samples in training set: {df_train["Heat"].sum()} / {len(df_train)}')
        print(f'Number of positive samples in validation set: {df_valid["Heat"].sum()} / {len(df_valid)}')
        dataset_kargs = dict(
            dict_device=self.setup_kargs['DICT_DEVICE'],
            modality=self.setup_kargs['MODALITY'],
            sensor_dir=self.setup_kargs['SENSOR_DIR'],
            sampling_rate=self.setup_kargs['SAMPLING_RATE']
        )
        self.dataset_train = HeatCattleDataset(df_train, mode='train', **dataset_kargs)
        self.dataset_valid = HeatCattleDataset(df_valid, mode='valid', **dataset_kargs)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=False, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=15)


class InferenceDataset(Dataset):
    def __init__(self, p_csv, stlide_minutes, sampling_rate):
        self.df = pd.read_csv(p_csv)
        self.df.columns = ['Timestamp', 'cattle_id', 'x', 'y', 'z']
        self.df = self.df.set_index('Timestamp')
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        self.df = self.df.groupby(self.df.index).mean()
        self.df = self.df.drop('cattle_id', axis=1)
        self.range_time = self.df.index[-1] - self.df.index[0]
        self.stlide_minutes = stlide_minutes
        self.sampling_rate = sampling_rate

    def __len__(self):
        return int(self.range_time.total_seconds() / (self.stlide_minutes * 60))

    def __getitem__(self, index):
        df_x = self.df.copy()
        start_timestamp = df_x.index[0]
        min_timestamp = start_timestamp + pd.Timedelta(minutes=self.stlide_minutes * index)
        max_timestamp = start_timestamp + pd.Timedelta(minutes=self.stlide_minutes * index) + pd.Timedelta(hours=2)
        mask = (df_x.index >= min_timestamp) & (df_x.index <= max_timestamp)
        df_x = df_x.loc[mask]
        min_q = 0.01
        max_q = 0.99
        quantiles = df_x[['x', 'y', 'z']].quantile([min_q, max_q])
        df_x = df_x[
            (df_x['x'].between(quantiles.loc[min_q, 'x'], quantiles.loc[max_q, 'x'])) &
            (df_x['y'].between(quantiles.loc[min_q, 'y'], quantiles.loc[max_q, 'y'])) &
            (df_x['z'].between(quantiles.loc[min_q, 'z'], quantiles.loc[max_q, 'z']))
        ]
        range_timestamp = pd.date_range(start=min_timestamp, end=max_timestamp, freq=f'{int(1000/self.sampling_rate)}ms')
        df_timestamp = pd.DataFrame(index=range_timestamp)
        df_timestamp[['x', 'y', 'z']] = 0
        df_x = df_x.add(df_timestamp, fill_value=np.nan)
        df_x = df_x.interpolate(method='time', limit_direction='both', limit=None)
        torch_x = torch.tensor(df_x.loc[:, ['x', 'y', 'z']].values, dtype=torch.float32).T
        return {
            'min_timestamp_int': int(time.mktime(min_timestamp.timetuple())),
            'max_timestamp_int': int(time.mktime(max_timestamp.timetuple())),
            'x': torch_x
        }
