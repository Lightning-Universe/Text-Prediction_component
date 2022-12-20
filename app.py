#! pip install -e . # change to install from github once public
#! ltt install --pytorch-channel nightly torch --upgrade git+https://github.com/Lightning-AI/lightning git+https://github.com/Lightning-AI/lightning-minGPT git+https://github.com/Lightning-AI/lightning-LLMs
#! curl https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt --create-dirs -o ${HOME}/data/shakespeare/input.txt -C -


import lightning as L
import os, torch
from lightning_mingpt import models
from lit_llms.tensorboard import DriveTensorBoardLogger, MultiNodeLightningTrainerWithTensorboard

from lai_textpred import default_callbacks, gpt_20b, WordDataset


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
        # CONFIGURE YOUR MODEL
        # --------------------
        model = models.FSDPGPT(
            vocab_size=train_dataset.vocab_size, block_size=int(train_dataset.block_size),
            model_type=None, **gpt_20b,
        )


        # -------------------
        # RUN YOUR FINETUNING
        # -------------------
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
