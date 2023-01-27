#! pip install git+https://github.com/Lightning-AI/lightning-LLMs git+https://github.com/Lightning-AI/LAI-Text-Prediction-Component
#! curl https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt --create-dirs -o ${HOME}/data/shakespeare/input.txt -C -

import glob
import json
import os
import sys
import threading
import time

import lightning as L
import psutil
import torch
from lightning_gpt import models
from lit_llms.tensorboard import (
    MultiNodeLightningTrainerWithTensorboard, DriveTensorBoardLogger,
)


from lai_textpred import (
    WordDataset,
    default_callbacks,
    error_if_local,
    gpt_1_7b,
    gpt_2_9b,
    gpt_4_4b,
    gpt_8_4b,
    gpt_10b,
    gpt_20b,
    gpt_45b,
)

class NetIOMonitor:
    def __init__(self, time_resolution, dump_at, file_path):
        self.time_resolution = time_resolution
        self.dump_at = dump_at
        self.vals = None
        self.file_path = file_path
        self.last_dump = 0

    def reset_vals(self):
        self.vals = []

    def dump_vals(self):
        print(f'dumping {len(self.vals)} values with a size of {sys.getsizeof(self.vals)/1024} MB')

        file_path = self.file_path.format(part=self.last_dump + 1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.vals, f)

        self.last_dump += 1

    @staticmethod
    def _get_vals():
        net_vals = psutil.net_io_counters()._asdict()
        for k in ('errin', 'errout', 'dropin', 'dropout'):
            net_vals.pop(k)

        return net_vals
    def __call__(self):
        self.reset_vals()
        while True:
            self.vals.append((time.time(), self._get_vals()))
            time.sleep(self.time_resolution)

            if len(self.vals) >= self.dump_at:
                self.dump_vals()
                self.reset_vals()

    @staticmethod
    def _get_id_from_filename(fname: str, split_sep: str = '-', keyword: str = 'rank'):
        fname = os.path.basename(fname)
        fname_splits = fname.split(split_sep)
        for split in fname_splits:
            if split.startswith(keyword):
                return int(split.replace(keyword, '').replace('.json', ''))
    @staticmethod
    def collect(filepath):
        files = sorted(glob.glob(os.path.join(os.path.dirname, filepath, '*.json')))
        all_ranks = set([NetIOMonitor._get_id_from_filename(f, '-', 'rank') for f in files])
        per_rank = {i: sorted([f for f in files if NetIOMonitor._get_id_from_filename(f, '-', 'rank') == i]) for i in all_ranks}

        per_rank_data = {i: [] for i in all_ranks}
        for rank, files in per_rank.items():
            for curr_file in files:
                with open(curr_file, 'r') as f:
                    per_rank_data[rank] += json.load(f)
        return {i: sorted(per_rank_data[i], key=lambda x: x[0]) for i in all_ranks} # sort by time

class NetIOCallback(L.pytorch.Callback):
    def __init__(self, time_resolution, dump_at, file_path):
        self.time_resolution = time_resolution
        self.dump_at = dump_at
        self.file_path = file_path
        self.monitor = None

    def on_train_start(self, trainer, pl_module):
        self.monitor = NetIOMonitor(self.time_resolution, self.dump_at, self.file_path.format(rank=trainer.global_rank))
        self.monitor.reset_vals()
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitor_thread.start()

    def on_train_end(self, trainer, pl_module):
        self.monitor_thread._stop()
        self.monitor_thread.join()


class WordPrediction(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        error_if_local()

        # -------------------
        # CONFIGURE YOUR DATA
        # -------------------
        with open(os.path.expanduser("~/data/shakespeare/input.txt")) as f:
            text = f.read()
        train_dataset = WordDataset(text, 5)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, num_workers=4, shuffle=True
        )

        # --------------------
        # CONFIGURE YOUR MODE
        # --------------------
        model = models.DeepSpeedMinGPT(
            vocab_size=train_dataset.vocab_size, block_size=int(train_dataset.block_size),
            fused_adam=False,  **gpt_20b,
        )

        # -----------------
        # RUN YOUR TRAINING
        # -----------------
        trainer = L.Trainer(
            max_epochs=2, limit_train_batches=250,
            precision=16, strategy="deepspeed_stage_3_offload",
            callbacks=default_callbacks(), log_every_n_steps=5,
            logger=DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
        )
        trainer.fit(model, train_loader)


app = L.LightningApp(
    MultiNodeLightningTrainerWithTensorboard(
        WordPrediction,
        num_nodes=3,
        cloud_compute=L.CloudCompute("gpu-fast-multi"),
    )
)
