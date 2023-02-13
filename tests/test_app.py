import logging
import sys
from urllib.request import urlretrieve

import lightning
import os

import pytest
from collections import namedtuple
import torch.nn
from app import WordPrediction
from lightning.app.utilities.tracer import Tracer
from lit_llms.tensorboard import (
    MultiNodeLightningTrainerWithTensorboard,
)
from lightning.app.runners import MultiProcessRuntime
from lightning_gpt.models import DeepSpeedMinGPT, MinGPT


from lightning.app.utilities.cloud import is_running_in_cloud

from lightning.app.testing import LightningTestApp

_DATA_DIR = os.path.expanduser('~/data/yelp')


class DummyPred(WordPrediction):
    def run(self):
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
        def trainer_pre_fn(trainer, *args, **kwargs):
            kwargs['max_epochs'] = 2
            kwargs['limit_train_batches'] = 2
            kwargs['limit_val_batches'] = 2
            if not is_running_in_cloud() and not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):

                kwargs['strategy'] = 'ddp'

            return {}, args, kwargs

        def multinode_pre_fn(multinode, *args, **kwargs):
            kwargs['num_nodes'] = 2
            return {}, args, kwargs

        def model_init_prefn(cls, *args, **kwargs):
            args = list(args)
            if args:
                args[0] = 'bigscience/bloom-560m'
            else:
                args.append('bigscience/bloom-560m')

            kwargs.pop('pretrained_model_name_or_path', None)
            kwargs['return_dict'] = True
            return {}, args, kwargs

        def lightningmodule_pre_fn(lm, **kwargs):
            kwargs['n_layer'] = None,
            kwargs['n_head'] = None,
            kwargs['n_embd']= None,
            kwargs['model_type'] = 'gpt-nano'

            return {}, (), kwargs

        def lightning_trainer_strategy_model_adapt(trainer: lightning.Trainer, model: DeepSpeedMinGPT, train_dataloader):
            if isinstance(trainer.strategy, (lightning.pytorch.strategies.DDPStrategy, lightning.pytorch.strategies.DDPSpawnStrategy)):
                model = MinGPT(vocab_size=model.hparams.vocab_size, block_size=model.hparams.block_size, model_type=model.hparams.model_type, n_head=model.hparams.n_head, n_layer=model.hparams.n_layer, n_embd=model.hparams.n_embd)

            return {}, (model, train_dataloader), {}

        tracer = Tracer()
        tracer.add_traced(lightning.Trainer, '__init__', pre_fn=trainer_pre_fn)
        tracer.add_traced(MultiNodeLightningTrainerWithTensorboard, '__init__', pre_fn=multinode_pre_fn)
        tracer.add_traced(DeepSpeedMinGPT, '__init__', pre_fn=lightningmodule_pre_fn)
        tracer.add_traced(lightning.Trainer, 'fit', pre_fn=lightning_trainer_strategy_model_adapt)

        tracer._instrument()
        ret_val = super().run()
        tracer._restore()
        return ret_val

def assert_logs(logs):
    expected_strings = [
        # don't include values for actual hardware availability as this may depend on environment.
        'GPU available: ',
        'All distributed processes registered.',
        '674 K    Trainable params\n0         Non - trainable params\n674 K    Total params\n2.699   Total estimated model params size(MB)',
        'Epoch 0:',
        '`Trainer.fit` stopped: `max_epochs=2` reached.',
        'Input text:Input text:\n summarize: ML Ops platforms come in many flavors from platforms that train models'
    ]
    for curr_str in expected_strings:
        assert curr_str in logs

# @pytest.mark.skipif(not bool(int(os.environ.get('SLOW_TEST', '0'))), reason='Skipping Slow Test by default')
def test_app_locally():
    os.makedirs(_DATA_DIR, exist_ok=True)
    if not os.path.isfile(os.path.join(_DATA_DIR,'input.txt')):
        urlretrieve('https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt', os.path.join(_DATA_DIR, 'input.txt'))


    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    app = LightningTestApp(
        # lightning.app.components.LightningTrainerMultiNode(
        MultiNodeLightningTrainerWithTensorboard(
            DummyPred, num_nodes=2, cloud_compute=lightning.CloudCompute("gpu-fast-multi", disk_size=50),
        )
    )
    runtime = lightning.app.runners.MultiProcessRuntime(app)
    runtime.dispatch(open_ui=False)
    # TODO: find a way to collect stdout and stderr outputs of multiprocessing to assert expected logs
    # logs = ...
    # assert_logs(logs)



