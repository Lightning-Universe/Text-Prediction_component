<div align="center">
    <h1>
        <img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/01/Group-15.png">
        <br>
        Train Large Text Prediction Models with Lightning
        </br>
    </h1>

<div align="center">

<p align="center">
  <a href="#run">Run</a> •
  <a href="https://www.lightning.ai/">Lightning AI</a> •
  <a href="https://lightning.ai/lightning-docs/">Docs</a>
</p>

[![ReadTheDocs](https://readthedocs.org/projects/pytorch-lightning/badge/?version=stable)](https://lightning.ai/lightning-docs/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://www.pytorchlightning.ai/community)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lightning/blob/master/LICENSE)

</div>
</div>

______________________________________________________________________

Use Lightning train large language model for text generation, 
with as many parameters as you want (up to billions!). 

You can do this:
* using multiple GPUs
* across multiple machines
* with the most advanced and efficient training techniques
* on your own data
* all without any infrastructure hassle! 

All handled easily with the [Lightning Apps framework](https://lightning.ai/lightning-docs/).

### Prediction Example
A typical example of text prediction could look like this:

_Prompt:_ `Please be aware of the`

_Prediction_: `situation`

## Run

To run paste the following code snippet in a file `app.py`:

```python
#! pip install light-the-torch
#! ltt install --upgrade git+https://github.com/Lightning-AI/lightning-LLMs git+https://github.com/Lightning-AI/LAI-Text-Prediction-Component
#! curl https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt --create-dirs -o ${HOME}/data/shakespeare/input.txt -C -


import lightning as L
import os, torch
from lightning_gpt import models
from lit_llms.tensorboard import DriveTensorBoardLogger, MultiNodeLightningTrainerWithTensorboard

from lai_textpred import default_callbacks, gpt_20b, WordDataset, error_if_local


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
            fused_adam=False, model_type=None, **gpt_20b,
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
```

### Running on the cloud

```bash
lightning run app app.py --cloud
```

Don't want to use the public cloud? Contact us at `product@lightning.ai` for early access to run on your private cluster (BYOC)!


### Running locally
This example is optimized for the cloud. 
It is therefore not possible to run this app locally. 
Please refer to [our text classification example](https://github.com/Lightning-AI/LAI-Text-Classification-Component) 
for a similar app to run locally.
