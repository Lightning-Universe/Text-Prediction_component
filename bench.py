#! pip install light-the-torch
#! pip install --upgrade git+https://github.com/Lightning-AI/lightning-LLMs git+https://github.com/Lightning-AI/LAI-Text-Prediction-Component
#! curl https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt --create-dirs -o ${HOME}/data/shakespeare/input.txt -C -


import os
from typing import Any, List, Mapping, Type

import lightning as L
import torch
from lightning.app.utilities.exceptions import ExitAppException
from lightning_gpt import models
from lit_llms.tensorboard import (
    DriveTensorBoardLogger,
    MultiNodeLightningTrainerWithTensorboard, TensorBoardWork,
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

class WordPrediction(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        error_if_local()
        # torch.backends.cudnn.deterministic = False
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.allow_tf32 = False
        # torch.backends.cuda.matmul.allow_tf32 = False

        # -------------------
        # CONFIGURE YOUR DATA
        # -------------------
        with open(os.path.expanduser("~/data/shakespeare/input.txt")) as f:
            text = f.read()
        train_dataset = WordDataset(text, 5)


        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=160, num_workers=8, shuffle=True, pin_memory=True
        )

        # --------------------
        # CONFIGURE YOUR MODE
        # --------------------
        model = models.DeepSpeedMinGPT(
            vocab_size=train_dataset.vocab_size,
            block_size=int(train_dataset.block_size),
            fused_adam=False,
            **gpt_4_4b,
        )

        # -----------------
        # RUN YOUR TRAINING
        # -----------------
        trainer = L.Trainer(
            max_epochs=2,
            limit_train_batches=25000,
            precision=16,
            strategy="deepspeed_stage_3_offload",
            callbacks=default_callbacks(detect_steady_state=True),
            log_every_n_steps=1,
            logger=DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
        )
        trainer.fit(model, train_loader)

class CustomMultiNodeLightningTrainerWithTensorboard(L.LightningFlow):
    def __init__(
        self,
        work_cls: Type[L.LightningWork],
        num_nodes: int,
        cloud_compute: L.CloudCompute,
        quit_tb_with_training: bool = True,
    ):
        super().__init__()
        tb_drive = L.app.storage.Drive("lit://tb_drive")

        self.multinode = L.app.components.LightningTrainerMultiNode(
            work_cls,
            num_nodes=num_nodes,
            cloud_compute=cloud_compute,
            tb_drive=tb_drive,
        )

        self.tb_drive = tb_drive

        self.tensorboard_work = TensorBoardWork(drive=self.tb_drive)

        self.quit_tb_with_training = quit_tb_with_training

    def run(self, *args: Any, **kwargs: Any) -> None:
        if self.quit_tb_with_training and self.tensorboard_work.is_running and all([w.has_succeeded for w in self.multinode.ws]):
            self.tensorboard_work.stop()
            raise ExitAppException

        self.multinode.run(*args, **kwargs)
        if any([w.is_running for w in self.multinode.ws]):
            self.tensorboard_work.run()

    def configure_layout(self) -> List[Mapping[str, str]]:
        return [{"name": "Training Logs", "content": self.tensorboard_work.url}]

app = L.LightningApp(
    CustomMultiNodeLightningTrainerWithTensorboard(
        WordPrediction,
        num_nodes=4,
        cloud_compute=L.CloudCompute("gpu-fast-multi", shm_size=2048),
    )
)
