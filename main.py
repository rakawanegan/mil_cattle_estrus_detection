import os
from glob import glob
from pathlib import Path
import json
import numpy as np
import pandas as pd
import umap
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
torch.set_float32_matmul_precision('medium')

from src.datamodule import HeatCattleDM, InferenceDataset
from src.model_interface import ModelInterface
from src.model import TransMIL
from src.utils import seed_everything, visualize_train_result, visualize_logits_umap


# Constants
BASE_DIR = Path('data/heat_dataset')
BASE_DIR = BASE_DIR.resolve()
INFERENCE_DIR = 'data/no_labeled_dataset'
SENSOR_DIR = os.path.join(BASE_DIR, 'sensor')
FILES_PATTERNS = {
    # 'approach': os.path.join(BASE_DIR, 'extract_heat_data2024-approach.csv'),
    'body_fluids': os.path.join(BASE_DIR, 'extract_heat_data2024-body-fuilds.csv'),
    'mounting': os.path.join(BASE_DIR, 'extract_heat_data2024-mounting.csv')
}
MODALITY = 'Acceleration'
SAMPLING_RATE = 10
USE_CHACE = True
INPUT_CH = 3
N_CLASSES = 1
PATCH_SIZE = 512
HIDDEN_DIM = 1024
TRAIN_RATIO = 0.8
EPOCHS = 10000
THRESHOLD = 0.7
STLIDE_MINUTES = 1
USE_COLS = ['Timestamp', 'Cattle_id', 'Interaction']
with open(os.path.join(BASE_DIR, 'device.json'), 'r') as file:
    DICT_DEVICE = json.load(file)
DICT_DEVICE = {int(k): v for k, v in DICT_DEVICE.items()}
INV_DICT_DEVICE = {v: k for k, v in DICT_DEVICE.items()}


def train():
    seed_everything(42)
    setup_kargs = dict(
        BASE_DIR=BASE_DIR,
        FILES_PATTERNS=FILES_PATTERNS,
        TRAIN_RATIO=TRAIN_RATIO,
        USE_COLS=USE_COLS,
        DICT_DEVICE=DICT_DEVICE,
        MODALITY=MODALITY,
        SENSOR_DIR=SENSOR_DIR,
        SAMPLING_RATE=SAMPLING_RATE,
        USE_CHACE=USE_CHACE,
    )
    dm = HeatCattleDM(setup_kargs)

    net = TransMIL(
        patch_size=PATCH_SIZE,
        input_ch=INPUT_CH,
        n_classes=N_CLASSES,
        hidden_dim=HIDDEN_DIM,
        threshold=THRESHOLD,
    )
    model = ModelInterface(net)

    logger = pl.loggers.CSVLogger(save_dir='logs/', name='estrus_detection_mil')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_BinaryPrecision',
        mode='max',
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        max_epochs=EPOCHS,
    )

    trainer.fit(
        model=model,
        datamodule=dm,
    )

    visualize_train_result(logger.log_dir)

    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()

    reduser = visualize_logits_umap(
        data_loader=train_loader,
        model=model,
        reducer=None,
        output_file=os.path.join(logger.log_dir, 'train_umap.png')
    )
    visualize_logits_umap(
        data_loader=valid_loader,
        model=model,
        reducer=reduser,
        output_file=os.path.join(logger.log_dir, 'valid_umap.png')
    )
    return logger.log_dir


def inference(log_dir):
    seed_everything(42)
    model_path = os.path.join(log_dir, 'checkpoints', 'last.ckpt')
    net = TransMIL(
        patch_size=PATCH_SIZE,
        input_ch=INPUT_CH,
        n_classes=N_CLASSES,
        hidden_dim=HIDDEN_DIM,
        threshold=THRESHOLD,
    )
    model = ModelInterface.load_from_checkpoint(
        checkpoint_path=model_path,
        net=net
    )
    model.eval()

    p_csv_list = glob(os.path.join(INFERENCE_DIR, '*Acc*.csv'))

    os.makedirs(os.path.join(log_dir, 'inference_results'), exist_ok=True)
    for p_csv in p_csv_list:
        print(f'Inference: {p_csv}')
        temp_df = pd.read_csv(p_csv)
        temp_df.columns = ['Timestamp', 'cattle_id', 'x', 'y', 'z']
        start_time = pd.to_datetime(temp_df['Timestamp'].iloc[0])
        end添削_time = pd.to_datetime(temp_df['Timestamp'].iloc[-1])
        test_dataset = InferenceDataset(p_csv, stlide_minutes=STLIDE_MINUTES, sampling_rate=SAMPLING_RATE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        per_result = list()
        for batch in test_loader:
            x = batch['x']
            min_timestamp = datetime.fromtimestamp(batch['min_timestamp_int'])
            max_timestamp = datetime.fromtimestamp(batch['max_timestamp_int'])
            x = x.to(model.device)
            results_dict = model(x)
            per_result.append(
                {
                    'min_timestamp': min_timestamp,
                    'max_timestamp': max_timestamp,
                    'Y_prob': results_dict['Y_prob'].detach().cpu().numpy().flatten()[0],
                }
            )
        df_result = pd.DataFrame(per_result)
        df_result['min_timestamp'] = df_result['min_timestamp']
        df_result = df_result.set_index('min_timestamp')
        df_result = pd.DataFrame(df_result['Y_prob'].rolling('15min').mean())
        df_result = df_result.reset_index()
        df_result.to_csv(os.path.join(log_dir, 'inference_results', f'{Path(p_csv).stem}.csv'), index=False)

    extract_date = lambda p: p.stem.split('_')[1]
    extract_device_id = lambda p: p.stem.split('_')[0][6:]

    for p_csv in glob(os.path.join(log_dir, 'inference_results', '*.csv')):
        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        date = extract_date(Path(p_csv))
        device_id = extract_device_id(Path(p_csv))
        cattle_id = INV_DICT_DEVICE[int(device_id)]
        df_result = pd.read_csv(p_csv)
        df_result.columns = ['timestamp', 'Y_prob']
        df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])
        df_result = df_result.sort_values('timestamp')
        df_result.plot.scatter(x='timestamp', y='Y_prob', s=1, c='blue', ax=ax)
        plt.title(f'Inference Results on {date}')
        plt.ylim(0, 1)
        plt.xticks(rotation=90)
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(log_dir, 'inference_results', f'date{date}-device_id{device_id}-cattle_id{cattle_id}.png'))
        plt.close()

    date_files = {}
    for p_csv in glob(os.path.join(log_dir, 'inference_results', '*.csv')):
        date = extract_date(Path(p_csv))
        if date not in date_files:
            date_files[date] = list()
        date_files[date].append(p_csv)

    for date, date_file in date_files.items():
        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        for p_csv in date_file:
            df_result = pd.read_csv(p_csv)
            df_result.columns = ['timestamp', 'Y_prob']
            df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])
            df_result = df_result.sort_values('timestamp')
            df_result.plot.scatter(x='timestamp', y='Y_prob', s=1, c='blue', ax=ax)
        plt.title(f'Inference Results on {date}')
        plt.ylim(0, 1)
        plt.xticks(rotation=90)
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.savefig(os.path.join(log_dir, 'inference_results', f'{date}.png'))
        plt.close()


def main():
    log_dir = train()
    inference(log_dir)


if __name__ == "__main__":
    main()
