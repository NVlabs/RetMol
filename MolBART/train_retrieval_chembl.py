# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for RetMol. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# coding=utf-8

import os
import sys
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
import deepspeed
from deepspeed.utils import RepeatingLoader

from utils.tokenizer import load_tokenizer
from megatron_molbart.decoder import DecodeSampler
from csv_data_retrieval import MoleculeDataLoader
from megatron_molbart.megatron_bart import MegatronBARTRetrieval
from megatron_molbart.util import DEFAULT_CHEM_TOKEN_START, DEFAULT_VOCAB_PATH, REGEX
from megatron_molbart.checkpointing import save_megatron_checkpoint, load_deepspeed_iteration, get_checkpoint_name

project_home = os.environ['PROJECT_HOME']
sys.path.insert(1, os.path.join(project_home, 'MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism'))
from megatron import print_rank_0, get_tensorboard_writer, get_timers, mpu, get_args
from megatron.initialize import initialize_megatron
from megatron.model.transformer import LayerNorm
from megatron.learning_rates import AnnealingLR
from megatron.utils import reduce_losses

tokenizer = load_tokenizer(vocab_path=DEFAULT_VOCAB_PATH, chem_token_start=DEFAULT_CHEM_TOKEN_START, regex=REGEX)
num_batches_processed = 0
epochs = 0


def get_params_for_weight_decay_optimization(module):
    """
    Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, LayerNorm):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


def get_deepspeed_checkpoint_dir(save_dir):
    return os.path.join(*os.path.split(save_dir)[:-1], 'deepspeed')


class RepeatingLoader:

    def __init__(self, loader):
        """
        Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.
        Args:
            loader (iterator): The data loader to repeat.
        """

        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        global epochs
        global num_batches_processed
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
            if torch.distributed.get_rank() == 0:
                epochs += 1
                num_batches_processed = 0
        return batch


def build_model(args):
    VOCAB_SIZE = len(tokenizer)
    MAX_SEQ_LEN = 512
    pad_token_idx = tokenizer.vocab[tokenizer.pad_token]
    sampler = DecodeSampler(tokenizer, MAX_SEQ_LEN)

    model = MegatronBARTRetrieval(
        sampler,
        pad_token_idx,
        VOCAB_SIZE,
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.hidden_size * 4,
        MAX_SEQ_LEN,
        dropout=0.1,
    )

    if args.train_from == 'stage1':
        model.add_fuser()  # attr fine tuning

    # load the encoder and decoder
    checkpoint_name = get_checkpoint_name(os.path.join(project_home, args.pretrain_model_path), args.model_ckpt_itr)

    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading checkpoint {}'.format(
            torch.distributed.get_rank(), checkpoint_name))
    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        # For backward compatibility.
        print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16.loss_scaler']
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
    except BaseException:
        print_rank_0('could not load the checkpoint')
        sys.exit()
    model.load_state_dict(state_dict['model'])

    if args.train_from == 'pretrain':
        model.add_fuser()  # attr fine tuning

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters()
                   if p.requires_grad)

    print_rank_0('Number of parameters in MegatronBART: '
                 + str(count_parameters(model)))
    return model


def get_optimizer(model, args):
    param_groups = get_params_for_weight_decay_optimization(model)
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False
    optimizer = AdamW(param_groups, lr=args.lr,
                      weight_decay=args.weight_decay,
                      betas=(args.adam_beta1, args.adam_beta2))
    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    # Add linear learning rate scheduler.
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup * args.train_iters,
        total_iters=args.train_iters,
        decay_style=args.lr_decay_style,
        min_lr=args.min_lr,
        last_iter=0,
        use_checkpoint_lr_scheduler=False,
        override_lr_scheduler=False,
    )

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = build_model(args)
    optimizer = get_optimizer(model.fuser, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    print_rank_0('DeepSpeed is enabled.')

    localrankmpi = int(os.getenv('LOCAL_RANK', '0'))
    rankmpi = int(os.getenv('RANK', '0'))
    args.rank = rankmpi
    args.local_rank = localrankmpi
    (model, optimizer, _, lr_scheduler) = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=(mpu if args.pipe_parallel_size == 0 else None),
        dist_init_required=False,
    )

    return (model, optimizer, lr_scheduler)


def get_batch(data_iterator):
    """Generate a batch"""

    global num_batches_processed
    keys = [
        'encoder_input',
        'encoder_pad_mask',
        'decoder_input',
        'decoder_pad_mask',
        'target',
        'target_pad_mask',
        'retrieved_smiles',
        'retrieved_pad_mask'
    ]
    datatype = torch.int64
    data = next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    encoder_tokens = data_b['encoder_input'].long()
    encoder_pad_mask = data_b['encoder_pad_mask'].bool()
    decoder_tokens = data_b['decoder_input'].long()
    decoder_pad_mask = data_b['decoder_pad_mask'].bool()
    target = data_b['target'].long()
    target_pad_mask = data_b['target_pad_mask'].long()
    retrieved_tokens = data_b['retrieved_smiles'].long()
    retrieved_pad_mask = data_b['retrieved_pad_mask'].bool()
    num_batches_processed += 1

    return {
        'encoder_input': encoder_tokens,
        'encoder_pad_mask': encoder_pad_mask,
        'decoder_input': decoder_tokens,
        'decoder_pad_mask': decoder_pad_mask,
        'target': target,
        'target_pad_mask': target_pad_mask,
        'retrieved_smiles': retrieved_tokens,
        'retrieved_pad_mask': retrieved_pad_mask,
    }


def forward_step(data_iterator, model):
    """Forward step."""

    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    batch = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model.
    outputs = model(batch)
    loss = model.module._calc_loss(batch, outputs)
    acc = model.module._calc_char_acc(batch, outputs)
    reduced_loss = reduce_losses([loss])
    reduced_acc = reduce_losses([acc])

    return (loss, {'mask loss': reduced_loss[0], 'acc': reduced_acc[0]})


def backward_step(optimizer, model, loss):
    """Backward step."""
    timers = get_timers()

    # Backward pass.
    timers('backward-backward').start()
    model.backward(loss)
    timers('backward-backward').stop()
    timers('backward-allreduce').reset()


def train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        lr_scheduler,
        pipe_parallel_size,
):
    """Single training step."""

    timers = get_timers()

    # Forward model for one step.
    timers('forward').start()
    (loss, loss_reduced) = forward_step_func(data_iterator, model)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, model, loss)
    timers('backward').stop()

    # Update parameters.
    timers('optimizer').start()
    model.step()
    timers('optimizer').stop()

    return loss_reduced


def train(
        forward_step_func,
        model,
        optimizer,
        lr_scheduler,
        train_data_iterator,
        trainloader,
        pipe_parallel_size,
        args,
):
    """Train the model function."""

    global num_batches_processed
    writer = get_tensorboard_writer()
    timers = get_timers()
    model.train()
    timers('interval time').start()

    while args.iteration < args.train_iters:
        loss = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            lr_scheduler,
            pipe_parallel_size,
        )

        args.iteration += 1
        print_rank_0('Iteration: ' + str(args.iteration) + '/'
                     + str(args.train_iters) + ', Loss: '
                     + str(loss['mask loss'].item()) + ', Acc: '
                     + str(loss['acc'].item()) + ', Num batches: '
                     + str(num_batches_processed) + '/'
                     + str(len(trainloader.loader)) + ', Epoch: '
                     + str(epochs))

        if torch.distributed.get_rank() == 0:
            writer.add_scalar('training mask loss', loss['mask loss'], args.iteration)
            writer.add_scalar('training acc', loss['acc'], args.iteration)

        # Checkpointing
        if args.iteration % args.save_interval == 0:
            # Deepspeed checkpoint
            path = get_deepspeed_checkpoint_dir(args.save)
            model.save_checkpoint(path)
            # Megatron checkpoint
            save_megatron_checkpoint(args.iteration, model, optimizer, lr_scheduler)

    return args.iteration


def run_training(ckpt_dir=None):
    deepspeed.init_distributed()
    initialize_megatron()
    args = get_args()
    args.iteration = 0

    pl.utilities.seed.seed_everything(args.seed)

    os.makedirs(args.save, exist_ok=True)
    if args.deepspeed:
        deepspeed_path = get_deepspeed_checkpoint_dir(args.save)
        os.makedirs(deepspeed_path, exist_ok=True)

    print_rank_0('Loading dataset(s) ...')
    loader = MoleculeDataLoader(args.dataset_path, args,
                                batch_size=args.batch_size, num_workers=32)
    train_dataloader = loader.get_data()

    print_rank_0('Setting up model ...')
    (model, optimizer, lr_scheduler) = setup_model_and_optimizer(args)

    if ckpt_dir is not None:
        path = get_deepspeed_checkpoint_dir(args.save) if args.deepspeed else args.save
        model.load_checkpoint(path)
        args.iteration = load_deepspeed_iteration(path)

    print_rank_0('Starting training ...')
    train_dataloader = RepeatingLoader(train_dataloader)

    train(
        forward_step,
        model,
        optimizer,
        lr_scheduler,
        iter(train_dataloader),
        train_dataloader,
        args.pipe_parallel_size,
        args,
    )


if __name__ == '__main__':
    run_training()
