#! pip install -e . # change to install from github once public
#! ltt install --pytorch-channel nightly torch --upgrade git+https://github.com/Lightning-AI/lightning git+https://github.com/Lightning-AI/lightning-minGPT
#! curl https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt --create-dirs -o ${HOME}/data/shakespeare/input.txt -C -
import os

import lightning as L
import mingpt.model
from lightning_mingpt.models import DeepSpeedGPT
from lightning_mingpt.data import CharDataset
import torch
from lai_charpred import default_callbacks, DriveTensorBoardLogger, Main, gpt_10b, gpt_20b


class MyDeepspeedGPT(DeepSpeedGPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mingpt = None

    def configure_sharded_model(self) -> None:
        self.mingpt = mingpt.model.GPT(self.mingpt_config)

    def configure_optimizers(self):
        return self.mingpt.configure_optimizers(self.mingpt_trainer_config)


class CharacterPrediction(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        with open(os.path.expanduser("~/data/shakespeare/input.txt")) as f:
            text = f.read()
        dataset = CharDataset(text, 50)
        train_dset, val_dset = torch.utils.data.random_split(
            dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=1, num_workers=4, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dset, batch_size=1, num_workers=4, shuffle=False
        )

        model = MyDeepspeedGPT(
            vocab_size=dataset.vocab_size,
            block_size=int(dataset.block_size),
            model_type=None,
            **gpt_10b,
            learning_rate=3e-4,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        trainer = L.Trainer(
            precision=16,
            # strategy='ddp',
            strategy="deepspeed_stage_3_offload",
            callbacks=default_callbacks(),
            logger=DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
            log_every_n_steps=5,
        )
        trainer.fit(model, train_loader, val_loader)


app = L.LightningApp(
    Main(
        CharacterPrediction,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu-fast-multi"),
    )
)
