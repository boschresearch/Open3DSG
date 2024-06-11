# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import random

import argparse
from datetime import datetime

import torch
import numpy as np

import pytorch_lightning as lightning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.strategies import DDPStrategy
from open3dsg.config.config import CONF
from open3dsg.scripts.trainer import D3SSGModule
from open3dsg.util.lightning_callbacks import LRLoggingCallback

if os.name == 'posix':
    import resource  # unix specific


def get_args():
    parser = argparse.ArgumentParser()

    # system params
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8, help="number of workers per gpu")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--run_name", type=str, help="dir name for tensorboard and checkpoints")
    parser.add_argument('--mixed_precision', action="store_true", help="Use mixed precision training")

    # optimizer params
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="reduce", help="lr_scheduler, options [cyclic, reduce]")
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-2)
    parser.add_argument("--bn_momentum", type=float, default=0.9, help="Initial batch norm momentum [default: 0.9]")
    parser.add_argument("--bn_decay", type=float, default=0.5,   help="Batch norm momentum decay gamma [default: 0.5]")
    parser.add_argument("--decay_step", type=float, default=1e5, help="Learning rate decay step [default: 20]",)

    parser.add_argument('--w_obj', type=float, default=1.0)
    parser.add_argument('--w_rel', type=float, default=1.0)

    # model params
    parser.add_argument('--use_rgb', action="store_true", help="Whether to use rgb features as input the the point net")
    parser.add_argument("--gnn_layers", type=int, default=4, help="number of gnn layers")
    parser.add_argument('--graph_backbone', default="message", nargs='?',
                        choices=['message', 'attention', 'transformer', 'mlp'])
    parser.add_argument('--gconv_dim', type=int, default=512, help='embedding dim for point features')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden dim for graph_convs')
    parser.add_argument('--max_nodes', type=int, default=9, help='max number of nodes in the graph')
    parser.add_argument('--max_edges', type=int, default=72,
                        help='max number of edges in the graph. Should correspond to n*(n-1) nodes')

    # data params
    parser.add_argument('--dataset', default='scannet', help="['scannet','3rscan']")
    parser.add_argument('--mini_dataset', action='store_true',
                        help="only load a tiny fraction of data for faster debugging")
    parser.add_argument('--augment', action="store_true",
                        help="use basic pcl augmentations that do not collide with scene graph properties")
    parser.add_argument("--top_k_frames", type=int, default=5, help="number of frames to consider for each instance")
    parser.add_argument("--scales", type=int, default=3, help="number of scales for each selected image")
    parser.add_argument('--dump_features', action="store_true", help="precompute 2d features and dump to disk")
    parser.add_argument('--load_features', default=None, help="path to precomputed 2d features")

    # model variations params
    parser.add_argument('--clip_model', default="OpenSeg", type=str,
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'OpenSeg'])
    parser.add_argument('--node_model', default='ViT-L/14@336px', type=str,
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--edge_model', default=None, type=str,
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--blip', action="store_true", help="Use blip for relation prediction")
    parser.add_argument('--avg_blip_emb', action='store_true', help="Average the blip embeddings across patches")
    parser.add_argument('--blip_proj_layers', type=int, default=3,
                        help="Number of projection layers to match blip embedding")
    parser.add_argument('--llava', action="store_true", help="Use llava for relation prediction")
    parser.add_argument('--avg_llava_emb', action="store_true", help="Average the llava embeddings across patches")
    parser.add_argument('--pointnet2', action="store_true",
                        help="Use pointnet++ for feature extraction. However RGB input not working")
    parser.add_argument("--clean_pointnet", action="store_true",
                        help="standard pretrained pointnet for feature extraction")
    parser.add_argument('--supervised_edges', action="store_true", help="Train edges supervised instead of open-vocab")

    # eval params
    parser.add_argument("--test", action="store_true", help="test the model")
    parser.add_argument("--checkpoint", type=str, help="Specify the checkpoint root", default=None)
    parser.add_argument('--weight_2d', type=float, default=0.5, help="2d-3d feature fusion weight")
    parser.add_argument('--n_beams', type=int, default=5, help="number of beams for beam search in LLM output")
    parser.add_argument('--gt_objects', action="store_true", help="Use GT objects for predicate prediction")
    parser.add_argument('--vis_graphs', action="store_true", help="save graph predictions to disk")
    parser.add_argument('--predict_materials', action="store_true",
                        help="predict materials from 3rscan seperate testset")
    parser.add_argument('--test_scans_3rscan', action="store_true",
                        help="test on 3rscan test set scans which are not labeled in 3dssg, it is needed for the material prediction")
    parser.add_argument('--predict_from_2d', action="store_true", help="predict only using 2d models")
    parser.add_argument('--quick_eval', action='store_true', help="only eval on a few samples")
    parser.add_argument('--object_context', action="store_true", help="prompt clip with: A [object] in a scene")
    parser.add_argument('--update_hparams', action="store_true", help="update hparams from checkpoint")
    parser.add_argument('--manual_mapping', action="store_true", help="Manually map some known predicates to GT")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # if os.name == 'posix':
    #     rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    #     resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

    if args.gpus == -1:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = args.gpus
    on_cluster = num_gpus > 1
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    hparams = vars(args)
    hparams['gpus'] = num_gpus
    if args.run_name is not None:
        hparams['run_name'] = args.run_name
    else:
        hparams['run_name'] = ''

    checkpoint: str = args.checkpoint

    is_test_or_tmp_run = "test" in hparams.get("run_name", None) or "tmp" in hparams.get("run_name", None)
    if args.test or is_test_or_tmp_run or hparams.get('mini_datatset', False):
        run_name = "tmp"
        logger = TensorBoardLogger(save_dir=CONF.PATH.TENSORBOARD, name=run_name, version=0)
    else:
        # generate a wandb like name for referenceing the runs
        adj = random.choice([line.rstrip().lower() for line in open(os.path.join(
            CONF.PATH.BASE, 'open3dsg', 'helpers', 'nature_adjectives.txt'), "r").readlines()])
        noun = random.choice([line.rstrip().lower() for line in open(os.path.join(
            CONF.PATH.BASE, 'open3dsg', 'helpers', 'nature_words.txt'), "r").readlines()])
        n = str(random.randint(1, 10000))
        run_name = '-'.join([adj, noun, n])
        hparams["run_name"] = run_name
        logger = MLFlowLogger(experiment_name="Open3DSG", run_name=run_name, tracking_uri=f"file:{CONF.PATH.MLFLOW}")
        print(logger.save_dir)
        print(logger.experiment_id)
        print(logger.run_id)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    lightning.seed_everything(args.seed)

    if checkpoint is not None:
        print("checkpoint:", checkpoint)

        if args.test or not args.update_hparams:
            model = D3SSGModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, map_location='cpu',
                                                     dataset=args.dataset,
                                                     supervised_edges=args.supervised_edges,
                                                     )
        else:
            model = D3SSGModule.load_from_checkpoint(checkpoint_path=checkpoint, strict=False,)

        if num_gpus == 1:
            glob_step = torch.load(checkpoint)['global_step']
            model.it = glob_step

        for k, v in hparams.items():
            model.hparams[k] = v

    else:
        model = D3SSGModule(hparams)
        model.it = 0

    print("Args: %s" % args)
    print("HParams: %s" % hparams)

    checkpoint_callback = lightning.callbacks.ModelCheckpoint(
        save_top_k=5,
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_last=True,
        #        every_n_train_steps=10000,
    )

    bnm_clip = 1e-2

    def bnm_lmbd(it):
        return max(args.bn_momentum * args.bn_decay ** (int(it * args.batch_size / args.decay_step)), bnm_clip,)

    precision = 32
    if hparams['mixed_precision']:
        if "A100" in torch.cuda.get_device_name(0):
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"

    hparams['start_time'] = datetime.now()
    trainer = lightning.Trainer(
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        devices=args.gpus, logger=logger,
        accelerator='auto',
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=precision,
        sync_batchnorm=True,
        callbacks=[
                    LRLoggingCallback(),
                    EarlyStopping(monitor="val/loss", mode="min", patience=10),
                    # GPUStatsMonitor(),
                    checkpoint_callback
        ],
        deterministic=False,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1 if args.mini_dataset else 100,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    if args.test is False:
        trainer.fit(model)
    else:
        if checkpoint is None:
            raise ValueError("Please provide a checkpoint to test")
        # import shutil
        # shutil.copyfile(checkpoint, '/'.join(checkpoint.split('/')[:-2])+f'/eval_{glob_step}.ckpt')
        model.eval()
        trainer.test(model)
