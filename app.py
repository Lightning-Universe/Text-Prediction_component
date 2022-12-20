#! pip install -e . # change to install from github once public
#! ltt install --pytorch-channel nightly torch --upgrade git+https://github.com/Lightning-AI/lightning git+https://github.com/Lightning-AI/lightning-minGPT git+https://github.com/Lightning-AI/lightning-LLMs
#! curl https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt --create-dirs -o ${HOME}/data/shakespeare/input.txt -C -


import lightning as L
import os, torch, deepspeed
from lightning_mingpt import models, data
from lit_llms.tensorboard import DriveTensorBoardLogger, MultiNodeLightningTrainerWithTensorboard

from lai_charpred import default_callbacks, gpt_10b, gpt_20b, gpt_45b


class MyDeepspeedGPT(models.DeepSpeedGPT):
    # def __init__(self, *args, activation_checkpointing: bool = False, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     if activation_checkpointing:
    #         self.forward = deepspeed.checkpointing.checkpoint(self.forward)
    def configure_optimizers(self):
        return self.mingpt.configure_optimizers(self.mingpt_trainer_config)


class CharacterPrediction(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        with open(os.path.expanduser("~/data/shakespeare/input.txt")) as f:
            text = f.read()
        dataset = data.CharDataset(text, 50)
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
            **gpt_20b,
            learning_rate=3e-4,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        # strategy = L.pytorch.strategies.DeepSpeedStrategy(stage=3,
        #     offload_optimizer=True,
        #     offload_parameters=True,
        #     partition_activations=True,
        #     cpu_checkpointing= False
        # )

        trainer = L.Trainer(
            precision=16,
            # strategy='ddp',
            strategy="deepspeed_stage_3_offload",
            # strategy=strategy,
            callbacks=default_callbacks(),
            logger=DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
            log_every_n_steps=5,
        )
        trainer.fit(model, train_loader, val_loader)


app = L.LightningApp(
    MultiNodeLightningTrainerWithTensorboard(
        CharacterPrediction,
        num_nodes=3,
        cloud_compute=L.CloudCompute("gpu-fast-multi"),
    )
)
