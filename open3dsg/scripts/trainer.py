# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import os
import time
import json
from datetime import datetime
import concurrent.futures
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import clip
import pytorch_lightning as lightning

from tqdm import tqdm

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from transformers import AutoModel
from graphviz import Digraph

from open3dsg.config.config import CONF
from open3dsg.data.open_dataset import Open2D3DSGDataset
from open3dsg.data.pcl_augmentations import PclAugmenter
from open3dsg.models.pointnet import feature_transform_reguliarzer
from open3dsg.models.sgpn import SGPN
from open3dsg.scripts.eval import get_eval, eval_attribute
from open3dsg.util.plotting_utils import *

REPORT_TEMPLATE_MAIN_EVAL = """
---------------------------------Report---------------------------------\n
[best] epoch: {epoch}\n
[sco.] Recall@1_object: {top1_recall_o}\n
[sco.] Recall@5_object: {top5_recall_o}\n
[sco.] Recall@10_object: {top10_recall_o}\n
[sco.] Recall@1_predicate: {top1_recall_p}\n
[sco.] Recall@3_predicate: {top3_recall_p}\n
[sco.] Recall@5_predicate: {top5_recall_p}\n
[sco.] Recall@1_relationship: {top1_rel}\n
[sco.] Recall@50_relationship: {top50_rel}\n
[sco.] Recall@100_relationship: {top100_rel}\n
--------------------------------------------------\n
[sco.] mRecall@1_object: {m_top1_recall_o}\n
[sco.] mRecall@5_object: {m_top5_recall_o}\n
[sco.] mRecall@10_object: {m_top10_recall_o}\n
[sco.] mRecall@1_predicate: {m_top1_recall_p}\n
[sco.] mRecall@3_predicate: {m_top3_recall_p}\n
[sco.] mRecall@5_predicate: {m_top5_recall_p}\n
[sco.] mRecall@50_relationship_by_pred: {m_top50_rel_by_pred}\n
[sco.] mRecall@100_relationship_by_pred: {m_top100_rel_by_pred}\n
[sco.] mRecall@50_relationship: {m_top50_rel}\n
[sco.] mRecall@100_relationship: {m_top100_rel}\n

"""

# node_color_list = ['aliceblue', 'antiquewhite', 'cornsilk3', 'lightpink', 'salmon', 'palegreen', 'khaki',
#                    'darkkhaki', 'orange']
colors = list(mcolors.TABLEAU_COLORS.keys())
node_color_list = list(mcolors.TABLEAU_COLORS.values())


query_colors = ["white", "black", "green", "blue", "red", "brown", "yellow", "gray",
                "orange", "purple", "pink", "beige", "bright", "dark", "light", "silver", "gold"]
query_materials = ['wooden', 'padded', 'glass', 'metal', 'ceramic', 'cardboard', 'plastic', 'carpet', 'stone', 'concrete']



def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


class D3SSGModule(lightning.LightningModule):
    def __init__(self, hparams):
        super(D3SSGModule, self).__init__()

        self.save_hyperparameters(hparams)

        self.model = SGPN(self.hparams)

        self.model.apply(inplace_relu)

        self.obj_class_dict = [line.rstrip() for line in open(os.path.join(CONF.PATH.R3SCAN_RAW, "classes.txt"), "r").readlines()]
        self.pred_class_dict = [line.rstrip() for line in open(os.path.join(
            CONF.PATH.R3SCAN_RAW, "relationships_custom.txt"), "r").readlines()]
        self.pred_class_dict_orig = [line.rstrip() for line in open(
            os.path.join(CONF.PATH.R3SCAN_RAW, "relationships.txt"), "r").readlines()]
        self.rel2idx = dict(zip(self.pred_class_dict_orig, range(len(self.pred_class_dict_orig))))
        self.known_mapping = dict(zip(self.pred_class_dict, self.pred_class_dict))
        self.known_mapping['to the left of'] = 'left of'
        self.known_mapping['to the right of'] = 'right of'
        self.known_mapping['next to'] = 'close by'  # 'none'
        self.known_mapping['above'] = 'higher than'  # 'none'
        self.known_mapping['under'] = 'lower than'
        self.known_mapping['placed on top'] = 'standing on'  # 'supported by'

        def map_rel2idx(class_name):
            return self.rel2idx.get(class_name, 0)
        self.rel2idx_mapping = np.vectorize(map_rel2idx)
        self.rel2rel_mapping = np.vectorize(self.known_mapping.get)
        self.cust_pred2pred = self.rel2idx_mapping(self.rel2rel_mapping(np.array(self.pred_class_dict)))
        self.cosSim1 = torch.nn.CosineSimilarity(dim=1)
        self.cosSim2 = torch.nn.CosineSimilarity(dim=2)

        self.transform = PclAugmenter(prob=0.25)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        path = CONF.PATH.HOME
        self.clip_path = os.path.join(CONF.PATH.FEATURES, f"clip_features_{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
        self.val_dataset = None
        self.train_dataset = None

    def setup(self, stage: str):
        def load_scan(base_path, file_path):
            return json.load(open(os.path.join(base_path, file_path)))["scans"]
        D3SSG_TRAIN = load_scan(CONF.PATH.R3SCAN_RAW, "3DSSG_subset/relationships_train.json")
        D3SSG_VAL = load_scan(CONF.PATH.R3SCAN_RAW, "3DSSG_subset/relationships_validation.json")
        D3SSG_TEST = load_scan(CONF.PATH.R3SCAN_RAW, "3DSSG_subset/relationships_test.json")

        SCANNET_TRAIN = load_scan(CONF.PATH.SCANNET, "subgraphs/relationships_train.json")
        SCANNET_VAL = load_scan(CONF.PATH.SCANNET, "subgraphs/relationships_validation.json")

        img_dim = 336 if self.hparams['clip_model'] == 'ViT-L/14@336px' else 224
        rel_img_dim = img_dim
        if self.hparams['edge_model']:
            rel_img_dim = 336 if self.hparams['edge_model'] == 'ViT-L/14@336px' else 224
        if self.hparams.get('dataset') == '3rscan':
            SCANNET_VAL, SCANNET_TRAIN = None, None
        else:
            D3SSG_VAL, D3SSG_TRAIN = None, None
        if stage == 'fit':
            if self.hparams.get('test_scans_3rscan'):
                print('Evaluating on 3RScan test set')
            self.val_dataset = Open2D3DSGDataset(
                relationships_R3SCAN=D3SSG_VAL if not self.hparams.get('test_scans_3rscan') else D3SSG_TEST,
                relationships_scannet=SCANNET_VAL,
                openseg=self.hparams['clip_model'] == 'OpenSeg',
                img_dim=img_dim,
                rel_img_dim=rel_img_dim,
                top_k_frames=self.hparams['top_k_frames'],
                scales=self.hparams['scales'],
                mini=self.hparams['mini_dataset'],
                load_features=self.hparams.get('load_features', None),
                blip=self.hparams.get('blip', False),
                llava=self.hparams.get('llava', False),
                half=self.hparams.get('quick_eval', False),
                max_objects=self.hparams.get('max_nodes', None),
                max_rels=self.hparams.get('max_edges', None)
            )
            self.train_dataset = Open2D3DSGDataset(
                relationships_R3SCAN=D3SSG_TRAIN,
                relationships_scannet=SCANNET_TRAIN,
                openseg=self.hparams['clip_model'] == 'OpenSeg',
                img_dim=img_dim,
                rel_img_dim=rel_img_dim,
                top_k_frames=self.hparams['top_k_frames'],
                scales=self.hparams['scales'],
                mini=self.hparams['mini_dataset'],
                load_features=self.hparams.get('load_features', None),
                blip=self.hparams.get('blip', False),
                llava=self.hparams.get('llava', False),
                max_objects=self.hparams.get('max_nodes', None),
                max_rels=self.hparams.get('max_edges', None)
            )

            # load pre-trained models
            if not self.hparams['clean_pointnet'] and not self.model.rgb and not self.model.nrm:
                self.model.load_pretained_cls_model(self.model.objPointNet)
                self.model.load_pretained_cls_model(self.model.relPointNet)

        elif stage == 'test':
            self.rel_mapper = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True).cuda()
            if self.hparams.get('test_scans_3rscan'):
                print('Evaluating on 3RScan test set')
            self.val_dataset = Open2D3DSGDataset(
                relationships_R3SCAN=D3SSG_VAL if not self.hparams.get('test_scans_3rscan') else D3SSG_TEST,
                relationships_scannet=SCANNET_VAL,
                openseg=self.hparams['clip_model'] == 'OpenSeg',
                img_dim=img_dim,
                rel_img_dim=rel_img_dim,
                top_k_frames=self.hparams['top_k_frames'],
                scales=self.hparams['scales'],
                mini=self.hparams['mini_dataset'],
                load_features=self.hparams.get('load_features', None),
                blip=self.hparams.get('blip', False),
                llava=self.hparams.get('llava', False),
                half=self.hparams.get('quick_eval', False)
            )

        if not self.hparams.get('dataset') == '3rscan':
            self.scannet_test_inst2label = {}
            test_scenes = set(s['scan'] for s in SCANNET_VAL)
            for scene in os.listdir(os.path.join(CONF.PATH.SCANNET, 'instance2labels')):
                if scene.split('inst')[0][:-1] in test_scenes:
                    with open(os.path.join(CONF.PATH.SCANNET, 'instance2labels', scene), 'r') as j:
                        self.scannet_test_inst2label[scene.split('inst')[0][:-1]] = np.vectorize(json.load(j).get)

        if self.hparams['test'] or not self.hparams['load_features']:
            if self.hparams['clip_model'] == "OpenSeg":
                self.model.OPENSEG = self.model.load_pretrained_clip_model(
                    target_model=self.model.OPENSEG, model=self.hparams["clip_model"])
            else:
                self.model.CLIP = self.model.load_pretrained_clip_model(target_model=self.model.CLIP, model=self.hparams["clip_model"])

            if self.hparams["clip_model"] != "OpenSeg":
                with torch.no_grad():
                    self.CLIP_NONE_EMB = F.normalize(self.model.CLIP.encode_text(clip.tokenize(['none']).to(self.model.clip_device)))
            if self.hparams['node_model']:
                self.model.CLIP_NODE = self.model.load_pretrained_clip_model(
                    target_model=self.model.CLIP_NODE, model=self.hparams["node_model"])
            if self.hparams['edge_model']:
                self.model.CLIP_EDGE = self.model.load_pretrained_clip_model(
                    target_model=self.model.CLIP_EDGE, model=self.hparams["edge_model"])
                with torch.no_grad():
                    self.CLIP_NONE_EMB = F.normalize(self.model.CLIP_EDGE.encode_text(clip.tokenize(['none']).to(self.model.clip_device)))

            if self.hparams.get('blip'):
                if self.hparams.get('dump_features'):
                    self.model.load_pretrained_blipvision_model()
                else:
                    self.model.load_pretrained_blip_model()
            elif self.hparams.get('llava'):
                self.model.load_pretrained_llava_model()

        else:
            if self.hparams.get('blip'):
                self.model.load_blip_pos_encoding()
            if self.hparams.get('llava'):
                self.model.load_pretrained_llava_model()

        # save params as metrics in mlflow for filtering
        if not self.hparams['test'] and type(self.logger) == lightning.loggers.MLFlowLogger:
            for hp_key, hp_val in self.hparams.items():
                if hp_key.startswith("Device"):
                    continue
                elif type(hp_val) in [float, int, bool]:
                    self.logger.experiment.log_metric(run_id=self.logger.run_id, key="param_"+hp_key, value=hp_val)

    def on_save_checkpoint(self, checkpoint):
        language_model_params = [key for key in checkpoint["state_dict"].keys() if (
            "BERT" in key or "CLIP" in key or "BLIP" in key or "LLaVA" in key or "OPENSEG" in key or key in 'rel_mapper')]

        for key in language_model_params:
            del checkpoint["state_dict"][key]

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], shuffle=True,
                                      collate_fn=self.train_dataset.collate_fn, num_workers=self.hparams['workers'], pin_memory=True)

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'], shuffle=False,
                                    collate_fn=self.val_dataset.collate_fn, num_workers=self.hparams['workers'], pin_memory=True)
        return val_dataloader

    def test_dataloader(self) -> DataLoader:
        test_dataloader = DataLoader(self.val_dataset, 1, shuffle=False,
                                     collate_fn=self.val_dataset.collate_fn, num_workers=self.hparams['workers'], pin_memory=True)
        return test_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams['lr'], weight_decay=1e-5)

        if self.hparams['lr_scheduler'] == 'cyclic':
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=self.hparams['lr']-(self.hparams['lr']/2),
                max_lr=self.hparams['lr']+(self.hparams['lr']/2),
                cycle_momentum=False,
                step_size_up=self.hparams['epochs']/6,
                step_size_down=self.hparams['epochs']/6,
                mode='triangular2'
            )
        elif self.hparams['lr_scheduler'] == 'cosine':
            lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                         first_cycle_steps=self.hparams['epochs']/3,
                                                         cycle_mult=1.0,
                                                         max_lr=self.hparams['lr']+(self.hparams['lr']/2),
                                                         min_lr=self.hparams['lr']-(self.hparams['lr']/2),
                                                         warmup_steps=self.hparams['epochs']/12,
                                                         gamma=0.7)
        elif self.hparams['lr_scheduler'] == 'cosine0':
            lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                         first_cycle_steps=self.hparams['epochs']/2,
                                                         cycle_mult=1.0,
                                                         max_lr=self.hparams['lr'],
                                                         min_lr=0,
                                                         warmup_steps=self.hparams['epochs']/12,
                                                         gamma=0.7)
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5 if not self.hparams['mini_dataset'] else 10000, threshold=0.0001, threshold_mode='rel',
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
            "interval": "step"
        }

    def on_after_batch_transfer(self, data_dict, dataloader_idx):

        if self.trainer.training and self.hparams.get('augment', False):
            data_dict = self.transform(data_dict)  # => we perform GPU/Batched data augmentation
        return data_dict

    def on_epoch_start(self) -> None:
        print("epoch {} starting...".format(self.current_epoch))

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if type(self.logger) == lightning.loggers.MLFlowLogger:
            self.logger.experiment.log_metric(run_id=self.logger.run_id, key="current_epoch", value=self.current_epoch)

    def on_train_batch_start(self, *args, **kwargs):
        pass

    def training_step(self, data_dict, *_, **__):

        data_dict = self._forward(data_dict)
        if self.hparams.get('dump_features'):
            self._dump_features(data_dict, data_dict["objects_id"].size(0), path=self.clip_path)
            return
        data_dict = self._compute_loss(data_dict)

        logs = {
            "train/loss": data_dict["loss"],
            "train/obj_loss": data_dict["obj_loss"],
            "train/rel_loss": data_dict["rel_loss"],
        }
        # self.log('tqdm_loss', data_dict["loss"], prog_bar=True, sync_dist=True)

        for k, v in logs.items():
            self.log(k, v, sync_dist=True, batch_size=self.hparams["batch_size"])  # , prog_bar=True if k=="train/loss" else False)

        return {'loss': data_dict["loss"]}

    def on_train_batch_end(self, *args, **kwargs):
        torch.cuda.empty_cache()  # maybe a problem wih accumulate gradients?

    def on_training_epoch_end(self):
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self) -> None:
        pass

    @torch.no_grad()
    def validation_step(self, data_dict, batch_ixd, dataloader_idx=0):

        data_dict = self._forward(data_dict)
        if self.hparams.get('dump_features'):
            self._dump_features(data_dict, data_dict["objects_id"].size(0), path=self.clip_path)
            return
        data_dict = self._compute_loss(data_dict)

        loss_dict = {
            "val/loss": data_dict["loss"],
            "val/obj_loss": data_dict["obj_loss"],
            "val/rel_loss": data_dict["rel_loss"],
        }

        for k, v in loss_dict.items():
            self.log(k, v, on_epoch=True, sync_dist=True, batch_size=1)  # , prog_bar=True if k=="val/loss" else False)

        return (data_dict, loss_dict)

    @torch.no_grad()
    def on_validation_epoch_end(self,):
        if self.hparams.get('dump_features'):
            exit(0)
        return

    def on_test_epoch_start(self) -> None:
        self.model.eval()

        self.test_step_outputs.clear()
        self.test_step_outputs = []

        self.test_pre_train_logs = {}
        self.test_pre_train_logs["objects_projection"] = []
        self.test_pre_train_logs["clip_obj_emb"] = []
        self.test_pre_train_logs["objects_cat"] = []
        self.test_pre_train_logs["predicates_projection"] = []
        self.test_pre_train_logs["clip_pred_emb"] = []
        self.test_pre_train_logs["predicate_strings"] = []
        self.test_pre_train_logs["edge_mapping"] = []
        self.scene_list = []
        self.loss_list = []
        self.recon_loss_list = []
        self.bbox_loss_list = []
        self.accuracy = {'total': [], 'left': [], 'right': [], 'front': [], 'behind': [],
                         'bigger': [], 'smaller': [], 'higher': [], 'lower': [], "same": []}
        self.start_t = time.time()

        self.rel_vocab = F.normalize(torch.from_numpy(self.rel_mapper.encode(self.pred_class_dict)), dim=-1).cuda()
        self.vis_dump_dir = CONF.PATH.BASE+'/vis_graphs/'+self.hparams['run_name'] + f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    @torch.no_grad()
    def test_step(self, data_dict, batch_ixd):
        assert data_dict['objects_id'].shape[0] == 1

        pred_dict = self._forward(data_dict)
        if self.hparams.get('dump_features'):
            self._dump_features(pred_dict, data_dict["objects_id"].size(0), path=self.clip_path)
            return
        vis = self.hparams.get('vis_graphs')

        if self.hparams['predict_from_2d']:
            objects_predict_clip, objects_probs_clip, object_mostlikely_clip, objects_valid_clip = self._predict_obj_from_clip(
                data_dict, data_dict['objects_id'].shape[0], query_classes=self.obj_class_dict, from_distill=False)
            if self.hparams.get('blip'):
                objects_clip, predicates_clip, relationships_clip, predicates_mapped_clip, predicates_mapped_probs_clip = self._predict_rel_from_blip(
                    data_dict, object_mostlikely_clip, objects_valid_clip, data_dict['objects_id'].shape[0], from_distill=False)
            elif self.hparams.get('llava'):
                objects_clip, predicates_clip, relationships_clip, predicates_mapped_clip, predicates_mapped_probs_clip = self._predict_rel_from_llava(
                    data_dict, object_mostlikely_clip, objects_valid_clip, data_dict['objects_id'].shape[0], from_distill=False)

        if not self.hparams['predict_from_2d'] or vis:
            objects_predict, objects_probs, object_mostlikely, objects_valid = self._predict_obj_from_clip(
                pred_dict, data_dict['objects_id'].shape[0], self.obj_class_dict, from_distill=True)
            if self.hparams.get('blip'):
                objects, predicates, relationships, predicates_mapped, predicates_mapped_probs = self._predict_rel_from_blip(
                    pred_dict, object_mostlikely, objects_valid, data_dict['objects_id'].shape[0], from_distill=True)
            elif self.hparams.get('llava'):
                objects, predicates, relationships, predicates_mapped, predicates_mapped_probs = self._predict_rel_from_llava(
                    pred_dict, object_mostlikely, objects_valid, data_dict['objects_id'].shape[0], from_distill=True)

        else:
            objects_predict, objects_probs, object_mostlikely, objects_valid = objects_predict_clip, objects_probs_clip, object_mostlikely_clip, objects_valid_clip
            objects, predicates, relationships, predicates_mapped, predicates_mapped_probs = objects_clip, predicates_clip, relationships_clip, predicates_mapped_clip, predicates_mapped_probs_clip

        if self.hparams.get('predict_materials'):
            materials_predict, materials_probs, materials_mostlikely, materials_valid = self._predict_obj_from_emb(
                pred_dict, data_dict['objects_id'].shape[0], query_materials)

        if self.hparams.get('predict_colors'):
            colors_predict, colors_probs, colors_mostlikely, colors_valid = self._predict_obj_from_emb(
                pred_dict, data_dict['objects_id'].shape[0], query_colors)

        if vis:
            if self.hparams.get('dataset') == '3rscan':
                dist = data_dict["predicate_min_dist"][0]
                dist_mask = torch.norm(dist, dim=-1) > 0.5
            else:
                dist = data_dict["predicate_dist"][0]
                dist_mask = torch.norm(dist, dim=-1) > 1.5
            dist_mask = dist_mask.cpu()[:data_dict["predicate_count"][0].item()]

            predicates = np.array(predicates[0])
            predicates[dist_mask] = 'none'
            predicates = [predicates.tolist()]

            predicates_clip = np.array(predicates_clip[0])
            predicates_clip[dist_mask] = 'none'
            predicates_clip = [predicates_clip.tolist()]

            predicates_mapped = np.array(predicates_mapped[0])
            predicates_mapped[dist_mask] = 'none'
            predicates_mapped = [predicates_mapped]

            predicates_mapped_clip = np.array(predicates_mapped_clip[0])
            predicates_mapped_clip[dist_mask] = 'none'
            predicates_mapped_clip = [predicates_mapped_clip]

            predicates_mapped = [list(predicates_mapped[0][:, 0])]
            predicates_mapped_clip = [list(predicates_mapped_clip[0][:, 0])]
            objects_gt = [objects[0][(data_dict['edges'] == i).cpu()[0][:data_dict['predicate_count'][0]]][0]
                          for i in torch.unique(data_dict['edges'][0])]
            self._vis_clip_graphs(data_dict, object_mostlikely, objects_gt, predicates, predicates_mapped,
                                  object_mostlikely_clip, predicates_clip, predicates_mapped_clip, dump_dir=self.vis_dump_dir)

        eval_dict = {}
        if self.hparams.get('dataset') == '3rscan':
            eval_dict['predicates_mapped_probs'] = predicates_mapped_probs
            eval_dict['predicates_mapped'] = predicates_mapped
            eval_dict['predicates_blip'] = predicates
            eval_dict['objects_predict'] = objects_predict
            eval_dict['objects_probs'] = objects_probs
            eval_dict['objects_predict_mostlikely'] = object_mostlikely
            eval_dict['scan_id'] = data_dict['scan_id']
            eval_dict["objects_id"] = data_dict["objects_id"].cpu()
            eval_dict["id2name"] = data_dict["id2name"]
            eval_dict["objects_count"] = data_dict["objects_count"].cpu()
            eval_dict["objects_cat"] = data_dict["objects_cat"].cpu()
            eval_dict["predicate_count"] = data_dict["predicate_count"].cpu()
            eval_dict["predicate_dist"] = data_dict["predicate_dist"].cpu()
            eval_dict["predicate_min_dist"] = data_dict["predicate_min_dist"].cpu()
            eval_dict["pairs"] = data_dict["pairs"].cpu()
            eval_dict["edges"] = data_dict["edges"].cpu()
            eval_dict["triples"] = data_dict["triples"]
            eval_dict["predicate_edges"] = data_dict['predicate_edges']
            eval_dict["relationships"] = (relationships, predicates, predicates_mapped, objects)

            if self.hparams.get('predict_materials'):
                eval_dict['materials_predict'] = materials_predict
                eval_dict['materials_probs'] = materials_probs
                eval_dict['materials_predict_mostlikely'] = materials_mostlikely

        self.test_step_outputs.append(eval_dict)
        return None

    @torch.no_grad()
    def on_test_epoch_end(self,):
        if not self.hparams.get('dataset') == '3rscan':
            return
        outputs = self.test_step_outputs
        relationship_analysis = {}
        relationship_analysis['relationships'] = []
        relationship_analysis['predicates'] = []
        relationship_analysis['gt'] = []
        relationship_analysis['top_choice'] = []
        relationship_analysis['mapped'] = []
        relationship_analysis['objects'] = []

        test_top1_o, test_top5_o, test_top10_o = [], [], []
        test_top1_p, test_top3_p, test_top5_p = [], [], []
        test_top1_rel, test_top10_rel, test_top50_rel, test_top100_rel = [], [], [], []

        test_top1_missed_o, test_top1_hit_o, test_top5_missed_o, test_top5_hit_o, test_top10_missed_o, test_top10_hit_o = [], [], [], [], [], []
        test_top1_missed_p, test_top1_hit_p, test_top3_missed_p, test_top3_hit_p, test_top5_missed_p, test_top5_hit_p = [], [], [], [], [], []

        eval_func = get_eval
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=8
        ) as executor:
            for res in list(tqdm(executor.map(eval_func, outputs), total=len(outputs), desc="Evaluating")):
                eval_dict = res
                test_top1_o.extend(eval_dict["top1_recall_o"])
                test_top5_o.extend(eval_dict["top5_recall_o"])
                test_top10_o.extend(eval_dict["top10_recall_o"])
                test_top1_p.extend(eval_dict['top1_recall_p'])
                test_top3_p.extend(eval_dict['top3_recall_p'])
                test_top5_p.extend(eval_dict['top5_recall_p'])
                test_top1_rel.extend(eval_dict['top1_recall_rel'])
                test_top50_rel.extend(eval_dict['top50_recall_rel'])
                test_top100_rel.extend(eval_dict['top100_recall_rel'])

                test_top1_hit_o.extend(eval_dict['top_1_hit_objects'])
                test_top1_missed_o.extend(eval_dict['top_1_missed_objects'])
                test_top5_hit_o.extend(eval_dict['top_5_hit_objects'])
                test_top5_missed_o.extend(eval_dict['top_5_missed_objects'])
                test_top10_hit_o.extend(eval_dict['top_10_hit_objects'])
                test_top10_missed_o.extend(eval_dict['top_10_missed_objects'])

                test_top1_missed_p.extend(eval_dict['top_1_missed_predicates'])
                test_top1_hit_p.extend(eval_dict['top_1_hit_predicates'])
                test_top3_missed_p.extend(eval_dict['top_3_missed_predicates'])
                test_top3_hit_p.extend(eval_dict['top_3_hit_predicates'])
                test_top5_missed_p.extend(eval_dict['top_5_missed_predicates'])
                test_top5_hit_p.extend(eval_dict['top_5_hit_predicates'])

        if self.hparams.get('predict_materials'):
            topk_attributes = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=8
            ) as executor:
                for res in list(tqdm(executor.map(eval_attribute, outputs), total=len(outputs), desc="Evaluating")):
                    eval_dict = res
                    topk_attributes.extend(eval_dict['topk_graph'])

            topk_attributes_by_class = [[item for sublist in lst for item in sublist] for lst in zip(*topk_attributes)]
            attribute_classwise_acc1 = [(np.array(topk) <= 1).sum()/len(topk) if len(topk) >
                                        0 else np.nan for topk in topk_attributes_by_class]
            attribute_classwise_acc3 = [(np.array(topk) <= 3).sum()/len(topk) if len(topk) >
                                        0 else np.nan for topk in topk_attributes_by_class]
            attribute_classwise_acc5 = [(np.array(topk) <= 5).sum()/len(topk) if len(topk) >
                                        0 else np.nan for topk in topk_attributes_by_class]
            print('Attributes Acc1', query_materials, attribute_classwise_acc1)
            print('Attributes Acc3', query_materials, attribute_classwise_acc3)
            print('Attributes Acc5', query_materials, attribute_classwise_acc5)

        test_top1_mrec_object = self.mRecall_objects(test_top1_hit_o, test_top1_missed_o)
        test_top5_mrec_object = self.mRecall_objects(test_top5_hit_o, test_top5_missed_o)
        test_top10_mrec_object = self.mRecall_objects(test_top10_hit_o, test_top10_missed_o)

        test_top1_mrec_predicate = self.mRecall_predicates(test_top1_hit_p, test_top1_missed_p)
        test_top3_mrec_predicate = self.mRecall_predicates(test_top3_hit_p, test_top3_missed_p)
        test_top5_mrec_predicate = self.mRecall_predicates(test_top5_hit_p, test_top5_missed_p)

        export = True
        if export:
            dump_dir = CONF.PATH.BASE + '/classwise_eval/'+self.hparams['run_name'] + f"_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
            os.makedirs(dump_dir, exist_ok=True)
            json.dump(test_top1_mrec_object, open(os.path.join(dump_dir, f"{'top1_mrec_object'}_eval_percent.json"), "w"))
            json.dump(test_top5_mrec_object, open(os.path.join(dump_dir, f"{'top5_mrec_object'}_eval_percent.json"), "w"))
            json.dump(test_top10_mrec_object, open(os.path.join(dump_dir, f"{'top10_mrec_object'}_eval_percent.json"), "w"))

            json.dump(test_top1_mrec_predicate, open(os.path.join(dump_dir, f"{'top1_mrec_predicate'}_eval_percent.json"), "w"))
            json.dump(test_top3_mrec_predicate, open(os.path.join(dump_dir, f"{'top3_mrec_predicate'}_eval_percent.json"), "w"))
            json.dump(test_top5_mrec_predicate, open(os.path.join(dump_dir, f"{'top5_mrec_predicate'}_eval_percent.json"), "w"))

        report = REPORT_TEMPLATE_MAIN_EVAL.format(
            epoch=self.current_epoch,
            top1_recall_o=round(np.average(test_top1_o), 5),
            top5_recall_o=round(np.average(test_top5_o), 5),
            top10_recall_o=round(np.average(test_top10_o), 5),
            top1_recall_p=round(np.nanmean(test_top1_p), 5),
            top3_recall_p=round(np.nanmean(test_top3_p), 5),
            top5_recall_p=round(np.nanmean(test_top5_p), 5),
            top1_rel=round(np.nanmean(test_top1_rel), 5),
            top50_rel=round(np.nanmean(test_top50_rel), 5),
            top100_rel=round(np.nanmean(test_top100_rel), 5),

            m_top1_recall_o=round(np.average(np.nanmean(np.array(list(test_top1_mrec_object.values())))), 5),
            m_top5_recall_o=round(np.average(np.nanmean(np.array(list(test_top5_mrec_object.values())))), 5),
            m_top10_recall_o=round(np.average(np.nanmean(np.array(list(test_top10_mrec_object.values())))), 5),
            m_top1_recall_p=round(np.average(np.nanmean(np.array(list(test_top1_mrec_predicate.values())))), 5),
            m_top3_recall_p=round(np.average(np.nanmean(np.array(list(test_top3_mrec_predicate.values())))), 5),
            m_top5_recall_p=round(np.average(np.nanmean(np.array(list(test_top5_mrec_predicate.values())))), 5),
            m_top50_rel_by_pred=0,
            m_top100_rel_by_pred=0,
            m_top50_rel=0,
            m_top100_rel=0,
        )

        print(report)
        with open(dump_dir+'/eval_metrics.txt', "w") as file:
            file.write(report)

        self.test_step_outputs.clear()
        return torch.tensor([1])

    def _forward(self, data_dict):
        data_dict = self.model(data_dict)
        return data_dict

    def _compute_loss(self, data_dict, mat_diff_loss_scale=0.001):
        batch_size = data_dict["objects_id"].size(0)

        if self.hparams.get('dataset') == '3rscan' and self.hparams.get('supervised_edges'):
            data_dict = self._supervised_dist_loss(batch_size, data_dict)
        else:
            data_dict = self._distillation_loss(batch_size, data_dict)

        mat_diff_loss = 0
        if not self.hparams.get('pointnet2', False):
            for it in data_dict["trans_feat"]:
                tmp = feature_transform_reguliarzer(it)
                mat_diff_loss = mat_diff_loss + tmp * mat_diff_loss_scale

            data_dict["L_mat_diff"] = mat_diff_loss
        else:
            data_dict["L_mat_diff"] = torch.FloatTensor([0]).to(data_dict["objects_id"].device)

        data_dict["loss"] = data_dict['loss'] + 0.1 * mat_diff_loss

        return data_dict

    def _mask_features(self, data_dict, clip_obj_emb, clip_rel_emb, bidx, obj_count, rel_count):
        obj_valids = None
        clip_rel_emb_masked = None
        if type(clip_obj_emb) is torch.Tensor:
            if self.hparams['clip_model'] == "OpenSeg":
                clip_obj2frame_mask = data_dict['obj2frame_raw_mask'][bidx][:obj_count]
            else:
                clip_obj2frame_mask = data_dict['obj2frame_mask'][bidx][:obj_count]

            clip_obj_mask = torch.arange(clip_obj_emb.size(1)).unsqueeze(0).to(
                clip_obj2frame_mask.device) < clip_obj2frame_mask.unsqueeze(1)
            clip_obj_emb[~clip_obj_mask] = np.nan
            clip_obj_emb = torch.nanmean(clip_obj_emb, dim=1)

            obj_valids = ~torch.isnan(clip_obj_emb).all(-1)

        if type(clip_rel_emb) is torch.Tensor:
            clip_rel2frame_mask = data_dict['rel2frame_mask'][bidx][:rel_count]
            clip_rel_mask = torch.arange(clip_rel_emb.size(1)).unsqueeze(0).to(
                clip_rel2frame_mask.device) < clip_rel2frame_mask.unsqueeze(1)
            clip_rel_emb[~clip_rel_mask] = np.nan
            clip_rel_emb = torch.nanmean(clip_rel_emb, dim=1)

            clip_rel_emb_masked = torch.zeros_like(clip_rel_emb)
            clip_rel_emb_masked[clip_rel2frame_mask > 0] = clip_rel_emb[clip_rel2frame_mask > 0]
            if not self.hparams.get('blip') and not self.hparams.get('llava'):
                clip_rel_emb_masked[clip_rel2frame_mask == 0] = self.CLIP_NONE_EMB.to(clip_rel_emb_masked[clip_rel2frame_mask == 0].dtype)
            else:
                clip_rel_emb_masked[clip_rel2frame_mask == 0] = np.nan

        return obj_valids, clip_obj_emb, clip_rel_emb_masked

    def _distillation_loss(self, batch_size, data_dict):
        obj_loss = []
        rel_loss = []
        for bidx in range(batch_size):
            obj_count = int(data_dict["objects_count"][bidx].item())
            rel_count = int(data_dict["predicate_count"][bidx].item())

            clip_obj_emb = data_dict['clip_obj_encoding'][bidx][:obj_count]
            clip_rel_emb = data_dict['clip_rel_encoding'][bidx][:rel_count]

            clip_obj_emb = clip_obj_emb/clip_obj_emb.norm(dim=-1, keepdim=True)

            if not self.hparams.get('load_features', None):
                obj_valids, clip_obj_emb, clip_rel_emb_masked = self._mask_features(
                    data_dict, clip_obj_emb, clip_rel_emb, bidx, obj_count, rel_count)
            else:
                obj_valids = data_dict['clip_obj_valids'][bidx][:obj_count]
                clip_rel_emb_masked = clip_rel_emb

            gcn_obj_emb = data_dict["objects_enc"][bidx][:obj_count]
            gcn_rel_emb = data_dict["predicates_enc"][bidx][:rel_count]

            if self.hparams['blip'] or self.hparams['llava']:
                if self.hparams['avg_blip_emb'] or self.hparams['avg_llava_emb']:
                    clip_rel_emb_masked = torch.mean(clip_rel_emb_masked, dim=-2)
                    nan_mask = ~torch.isnan(clip_rel_emb_masked).all(dim=-1)
                else:
                    if self.hparams['blip']:
                        nan_mask = ~torch.isnan(clip_rel_emb_masked).all(dim=-1).all(dim=-1)
                    elif self.hparams['llava']:
                        clip_rel_emb_masked = clip_rel_emb_masked.view(clip_rel_emb_masked.shape[0], -1)
                        nan_mask = ~torch.isnan(clip_rel_emb_masked).all(dim=-1)
            else:
                nan_mask = torch.ones(gcn_rel_emb.shape[0], dtype=True)

            if (~obj_valids).all():
                obj_loss_i = torch.tensor(0, dtype=torch.float).to(clip_obj_emb.device)
            else:
                obj_loss_i = (1 - self.cosSim1(gcn_obj_emb[obj_valids], clip_obj_emb[obj_valids])).mean()
            if (~nan_mask).all():
                rel_loss_i = torch.tensor(0, dtype=torch.float).to(obj_loss_i.device)
            else:
                rels = gcn_rel_emb[nan_mask].shape[0]
                # blip features that do not exists are nan
                rel_loss_i = (1 - self.cosSim1(gcn_rel_emb[nan_mask].view(rels, -1), clip_rel_emb_masked[nan_mask].view(rels, -1))).mean()

            obj_loss.append(obj_loss_i)
            rel_loss.append(rel_loss_i)

        data_dict['obj_loss'] = torch.stack(obj_loss).mean()
        data_dict['rel_loss'] = torch.stack(rel_loss).mean()
        data_dict['loss'] = self.hparams.get("w_obj", 1.0) * data_dict['obj_loss'] + self.hparams.get("w_rel", 1.0) * data_dict['rel_loss']
        data_dict['obj_loss'] = data_dict['obj_loss'].detach()
        data_dict['rel_loss'] = data_dict['rel_loss'].detach()
        return data_dict

    def _supervised_dist_loss(self, batch_size, data_dict):
        obj_loss = []
        rel_loss = []
        for bidx in range(batch_size):
            obj_count = int(data_dict["objects_count"][bidx].item())
            rel_count = int(data_dict["predicate_count"][bidx].item())

            clip_obj_emb = data_dict['clip_obj_encoding'][bidx][:obj_count]
            rel_classes = data_dict["predicate_cat"][bidx][:rel_count]

            clip_obj_emb = clip_obj_emb/clip_obj_emb.norm(dim=-1, keepdim=True)

            if not self.hparams.get('load_features', None):
                obj_valids, clip_obj_emb, _ = self._mask_features(data_dict, clip_obj_emb, None, bidx, obj_count, None)
            else:
                obj_valids = data_dict['clip_obj_valids'][bidx][:obj_count]

            gcn_obj_emb = data_dict["objects_enc"][bidx][:obj_count]
            rel_prediction = data_dict["rel_prediction"][bidx][:rel_count]

            if (~obj_valids).all():
                obj_loss_i = torch.tensor(0, dtype=torch.float).to(clip_obj_emb.device)
            else:
                obj_loss_i = (1 - self.cosSim1(gcn_obj_emb[obj_valids], clip_obj_emb[obj_valids])).mean()

            rel_loss_i = self.rel_supervision(rel_prediction, rel_classes.to(torch.float32))

            obj_loss.append(obj_loss_i)
            rel_loss.append(rel_loss_i)

        data_dict['obj_loss'] = torch.stack(obj_loss).mean()
        data_dict['rel_loss'] = torch.stack(rel_loss).mean()
        data_dict['loss'] = self.hparams.get("w_obj", 1.0) * data_dict['obj_loss'] + self.hparams.get("w_rel", 1.0) * data_dict['rel_loss']
        data_dict['obj_loss'] = data_dict['obj_loss'].detach()
        data_dict['rel_loss'] = data_dict['rel_loss'].detach()
        return data_dict

    def _dump_features(self, data_dict, batch_size, path=CONF.PATH.FEATURES):
        for bidx in range(batch_size):
            obj_count = int(data_dict["objects_count"][bidx].item())
            rel_count = int(data_dict["predicate_count"][bidx].item())

            clip_obj_emb = data_dict['clip_obj_encoding'][bidx][:obj_count]
            if not self.hparams.get('blip') and not self.hparams.get('llava'):
                clip_rel_emb = data_dict['clip_rel_encoding'][bidx][:rel_count]
            else:
                clip_rel_emb = data_dict['clip_rel_encoding'][bidx][:rel_count]

            obj_valids, clip_obj_emb, clip_rel_emb_masked = self._mask_features(
                data_dict, clip_obj_emb, clip_rel_emb, bidx, obj_count, rel_count)

            obj_clip_model = self.hparams['node_model'] if self.hparams['node_model'] and self.hparams['clip_model'] != "OpenSeg" else self.hparams['clip_model']
            rel_clip_model = self.hparams['edge_model'] if self.hparams['edge_model'] else self.hparams['clip_model']
            if self.hparams['blip']:
                rel_clip_model = "BLIP"
            elif self.hparams['llava']:
                rel_clip_model = "LLaVa"

            obj_path = os.path.join(path, 'export_obj_clip_emb_clip_' + obj_clip_model.replace('/', '-')+'_Topk_' + str(self.hparams['top_k_frames'])+'_scales_'+str(
                self.hparams['scales'])+'_vis_crit_' + str(self.val_dataset.obj_vis_crit)+'_vis_crit_mask_' + str(self.val_dataset.obj_mask_crit))
            obj_valid_path = os.path.join(path, 'export_obj_clip_valids')
            rel_path = os.path.join(path, 'export_rel_clip_emb_clip_' + rel_clip_model.replace('/', '-')+'_Topk_' + str(
                self.hparams['top_k_frames'])+'_scales_'+str(self.hparams['scales'])+'_vis_crit_' + str(self.val_dataset.rel_vis_crit))
            os.makedirs(obj_path, exist_ok=True)
            os.makedirs(obj_valid_path, exist_ok=True)
            os.makedirs(rel_path, exist_ok=True)

            torch.save(clip_obj_emb.detach().cpu(), os.path.join(obj_path, data_dict['scan_id'][bidx]+'.pt'))
            torch.save(obj_valids.detach().cpu(), os.path.join(obj_valid_path, data_dict['scan_id'][bidx]+'.pt'))
            torch.save(clip_rel_emb_masked.detach().cpu(), os.path.join(rel_path, data_dict['scan_id'][bidx]+'.pt'))

    @torch.no_grad()
    def _predict_obj_from_clip(self, data_dict, batch_size, query_classes=None, from_distill=True):
        objs_predict = []
        objects_probs = []
        objs_mostlikly = []
        objs_valid_batch = []
        for bidx in range(batch_size):
            obj_count = int(data_dict["objects_count"][bidx].item())
            clip_obj_emb = data_dict['clip_obj_encoding'][bidx][:obj_count]
            if from_distill:
                graph_obj_emb = data_dict['objects_enc'][bidx][:obj_count]

            if not self.hparams.get('load_features', None):
                obj_valids, clip_obj_emb, _ = self._mask_features(data_dict, clip_obj_emb, None, bidx, obj_count, None)
            else:
                obj_valids = data_dict['clip_obj_valids'][bidx][:obj_count]

            if from_distill:
                clip_obj_emb = clip_obj_emb.to(graph_obj_emb.dtype)
                clip_obj_emb[torch.isnan(clip_obj_emb)] = graph_obj_emb[torch.isnan(clip_obj_emb)]
                clip_obj_emb = self.hparams.get('weight_2d', 0.5)*clip_obj_emb+(1-self.hparams.get('weight_2d', 0.5))*graph_obj_emb

            _, objs_probs = self.model.predict_nodes(clip_obj_emb.float(), query_classes)
            _, objs = torch.sort(objs_probs, dim=-1, descending=True)
            objs_probs, objs = objs_probs.detach().cpu(), objs[:, :20].detach().cpu()
            objs_predict.append(objs)
            objects_probs.append(objs_probs)
            objs_mostlikly.append(objs[:, 0])
            objs_valid_batch.append(obj_valids)
        return objs_predict, objects_probs, objs_mostlikly, objs_valid_batch

    @torch.no_grad()
    def _predict_rel_from_blip(self, data_dict, obj_predict, obj_valids, batch_size, from_distill=True):
        relationships = []
        predicates = []
        predicates_mapped = []
        predicates_mapped_probs = []
        objects = []
        pred_names = np.array(self.pred_class_dict_orig)
        objects_gt = data_dict['objects_cat']
        for bidx in range(batch_size):
            rel_count = int(data_dict["predicate_count"][bidx].item())
            obj_count = int(data_dict['objects_count'][bidx].item())
            clip_rel_emb = data_dict['clip_rel_encoding'][bidx][:rel_count]

            edges = data_dict['edges'][bidx][:rel_count]

            if self.hparams.get('gt_objects') and not self.hparams.get('dataset') == '3rscan':
                object_edges = data_dict['objects_id'][bidx][:obj_count][edges].cpu().to(torch.long)
            elif self.hparams.get('gt_objects'):
                object_edges = objects_gt[bidx][:obj_count][edges].cpu().to(torch.long)
            else:
                object_edges = obj_predict[bidx][edges.cpu()]

            if self.hparams.get('dataset') == '3rscan':
                object_edges = np.array(self.obj_class_dict)[object_edges]
                object_edges[object_edges == 'socket'] = 'wall'
            else:
                inst2name = self.scannet_test_inst2label[data_dict['scan_id'][0].split('-')[0]]
                object_edges = inst2name(object_edges.numpy().astype(str))
                object_edges[object_edges == None] = 'object'

            if from_distill:
                graph_rel_emb = data_dict['predicates_enc'][bidx][:rel_count]

                clip_rel_emb_masked = clip_rel_emb
                if self.hparams.get('avg_blip_emb'):
                    graph_rel_emb = graph_rel_emb.unsqueeze(-2).repeat(1, clip_rel_emb.shape[-2], 1)

                clip_rel_emb_masked = clip_rel_emb_masked.to(graph_rel_emb.dtype)
                none_mask = torch.isnan(clip_rel_emb_masked)
                clip_rel_emb_masked[none_mask] = graph_rel_emb[none_mask]
                clip_rel_emb_masked = self.hparams.get('weight_2d', 0.5)*graph_rel_emb + \
                    (1-self.hparams.get('weight_2d', 0.5))*clip_rel_emb_masked
            else:
                if not self.hparams.get('load_features', None):
                    _, _, clip_rel_emb_masked = self._mask_features(data_dict, None, clip_rel_emb, bidx, None, rel_count)
                else:
                    clip_rel_emb_masked = clip_rel_emb
                none_mask = torch.isnan(clip_rel_emb_masked).all(dim=-1).all(dim=-1)

            predicate_prediction, relationship_prediction = self.model.blip_predict_relationship(clip_rel_emb_masked, object_edges)

            if self.hparams.get('mask_non_shared'):
                predicate_prediction[none_mask] = 'and'

            predicate_prediction_emb = F.normalize(torch.from_numpy(self.rel_mapper.encode(predicate_prediction)), dim=-1).cuda()
            predicate_mapping = (self.rel_vocab@predicate_prediction_emb.T)

            scatter_idx = torch.from_numpy(self.cust_pred2pred).view(-1, 1).expand(predicate_mapping.size()).to(predicate_mapping.device)
            agg_values = torch.zeros(predicate_mapping.size()).to(predicate_mapping.device).to(predicate_mapping.dtype)
            agg_values = agg_values.scatter_reduce(0, scatter_idx, predicate_mapping, reduce='amax')
            predicate_mapping = agg_values[:len(np.unique(self.cust_pred2pred))].T

            _, top_predictions = torch.sort(predicate_mapping, dim=1, descending=True)
            predicate_mapping, top_predictions = predicate_mapping.detach().cpu(), top_predictions.detach().cpu()
            assert predicate_mapping.shape[-1] == 27

            predicates.append(predicate_prediction)
            objects.append(object_edges), predicates_mapped.append(
                pred_names[top_predictions.cpu()]), predicates_mapped_probs.append(predicate_mapping)
            relationships.append(relationship_prediction)
        return objects, predicates, relationships, predicates_mapped, predicates_mapped_probs

    @torch.no_grad()
    def _predict_rel_from_llava(self, data_dict, obj_predict, obj_valids, batch_size, from_distill=True):
        relationships = []
        predicates = []
        predicates_mapped = []
        predicates_mapped_probs = []
        objects = []
        pred_names = np.array(self.pred_class_dict_orig)
        objects_gt = data_dict['objects_cat']
        for bidx in range(batch_size):
            rel_count = int(data_dict["predicate_count"][bidx].item())
            obj_count = int(data_dict['objects_count'][bidx].item())
            clip_rel_emb = data_dict['clip_rel_encoding'][bidx][:rel_count]

            edges = data_dict['edges'][bidx][:rel_count]
            if self.hparams.get('gt_objects') and not self.hparams.get('dataset') == '3rscan':
                object_edges = data_dict['objects_id'][bidx][:obj_count][edges].cpu().to(torch.long)
            elif self.hparams.get('gt_objects'):
                object_edges = objects_gt[bidx][:obj_count][edges].cpu().to(torch.long)
            else:
                object_edges = obj_predict[bidx][edges.cpu()]

            if self.hparams.get('dataset') == '3rscan':
                object_edges = np.array(self.obj_class_dict)[object_edges]
                object_edges[object_edges == 'socket'] = 'wall'
            else:
                inst2name = self.scannet_test_inst2label[data_dict['scan_id'][0].split('-')[0]]
                object_edges = inst2name(object_edges.numpy().astype(str))
                object_edges[object_edges == None] = 'object'

            if from_distill:
                graph_rel_emb = data_dict['predicates_enc'][bidx][:rel_count]
                clip_rel_emb_masked = clip_rel_emb
                if self.hparams.get('avg_llava_emb'):
                    graph_rel_emb = graph_rel_emb.unsqueeze(-2).repeat(1, clip_rel_emb.shape[-2], 1)

                clip_rel_emb_masked = clip_rel_emb_masked.to(graph_rel_emb.dtype)
                none_mask = torch.isnan(clip_rel_emb_masked)
                clip_rel_emb_masked[none_mask] = graph_rel_emb[none_mask]
            else:
                if not self.hparams.get('load_features', None):
                    _, _, clip_rel_emb_masked = self._mask_features(data_dict, None, clip_rel_emb, bidx, None, rel_count)
                else:
                    clip_rel_emb_masked = clip_rel_emb
                none_mask = torch.isnan(clip_rel_emb_masked).all(dim=-1).all(dim=-1)

            predicate_prediction, relationship_prediction = self.model.llava_predict_relationship(clip_rel_emb_masked, object_edges)
            if from_distill:
                if self.hparams.get('mask_non_shared'):
                    predicate_prediction[none_mask] = 'and'

            predicate_prediction_emb = F.normalize(torch.from_numpy(self.rel_mapper.encode(predicate_prediction)), dim=-1).cuda()
            predicate_mapping = (self.rel_vocab@predicate_prediction_emb.T)

            scatter_idx = torch.from_numpy(self.cust_pred2pred).view(-1, 1).expand(predicate_mapping.size()).to(predicate_mapping.device)
            agg_values = torch.zeros(predicate_mapping.size()).to(predicate_mapping.device).to(predicate_mapping.dtype)
            agg_values = agg_values.scatter_reduce(0, scatter_idx, predicate_mapping, reduce='amax')
            predicate_mapping = agg_values[:len(np.unique(self.cust_pred2pred))].T

            _, top_predictions = torch.sort(predicate_mapping, dim=1, descending=True)
            predicate_mapping, top_predictions = predicate_mapping.detach().cpu(), top_predictions.detach().cpu()
            assert predicate_mapping.shape[-1] == 27

            predicates.append(predicate_prediction)
            objects.append(object_edges), predicates_mapped.append(
                pred_names[top_predictions.cpu()]), predicates_mapped_probs.append(predicate_mapping)
            relationships.append(relationship_prediction)
        return objects, predicates, relationships, predicates_mapped, predicates_mapped_probs

    def _vis_clip_graphs(self, data_dict, objects_predict, objects_gt, predicates, predicates_mapped, objects_predict_clip, predicates_clip, predicates_mapped_clip, dump_dir='vis_graphs'):
        show_imgs = True & (self.hparams['load_features'] is None)
        print(data_dict['scan_id'][0])
        print(data_dict['objects_id'][0])

        self._vis_graph_scannet_clip(data_dict['scan_id'][0], objects_predict[0], objects_gt, predicates[0], data_dict['edges'][0][:len(
            predicates[0])], data_dict['objects_id'][0], filename=dump_dir+f"/{data_dict['scan_id'][0]}/graph")
        self._vis_graph_scannet_clip(data_dict['scan_id'][0], objects_predict[0], objects_gt, predicates_mapped[0], data_dict['edges'][0][:len(
            predicates_mapped[0])], data_dict['objects_id'][0], filename=dump_dir+f"/{data_dict['scan_id'][0]}/graph_predicates_mapped")

        if self.hparams['predict_from_2d']:
            self._vis_graph_scannet_clip(data_dict['scan_id'][0], objects_predict_clip[0], objects_gt, predicates_clip[0], data_dict['edges'][0][:len(
                predicates_clip[0])], data_dict['objects_id'][0], filename=dump_dir+f"/{data_dict['scan_id'][0]}/graph_clip")
            self._vis_graph_scannet_clip(data_dict['scan_id'][0], objects_predict_clip[0], objects_gt, predicates_mapped_clip[0], data_dict['edges'][0][:len(
                predicates_mapped_clip[0])], data_dict['objects_id'][0], filename=dump_dir+f"/{data_dict['scan_id'][0]}/graph_predicates_mapped_clip")

        if show_imgs:
            obj_imgs = data_dict['object_imgs'][0][:, 0][:data_dict['objects_count'][0]].cpu()
            if self.hparams['blip']:
                pred_imgs = [torch.from_numpy(np.array(img)/255).permute(2, 0, 1)
                             for img in np.array(data_dict['blip_images'][0])[:, 0][:data_dict['predicate_count'][0]]]
            else:

                pred_imgs = data_dict['relationship_imgs'][0][:, 0][:data_dict['predicate_count'][0]].cpu()

            # plot_3Dscene_imgs(shapes,bboxes,corners,obj_imgs)
            # plot_image9x9(pred_imgs)

        # plot_3Dscene(shapes,bboxes,corners)

        # plt.show()

    def _vis_graph_scannet_clip(self, scan_id, objects, objects_gt, predicates, edges, object_ids, filename='graph'):
        dot = Digraph(comment='The Scene Graph')
        dot.attr(rankdir='TB')

        dot.attr(label=scan_id)
        dot.attr('node', shape='oval', fontname='Sans')
        a = scan_id
        a, b = '-'.join(a.split('-')[:-1]), a.split('-')[-1]
        g_colors = node_color_list  # {o['id']: o['ply_color'] for o in self.scene_graphs_val[a+'_'+b]['objects']}
        for index in range(len(objects)):
            id = str(index)
            dot.attr('node', fillcolor=g_colors[index], style='filled')
            pred = self.obj_class_dict[objects[index]]
            pred = pred if pred != 'socket' else 'wall'
            dot.node(id, pred+f" [{objects_gt[index]}] "+'-'+str(object_ids[index].item()))

        # edges
        dot.attr('edge', fontname='Sans', color='black', style='filled')
        for i, edge in enumerate(edges):
            s, o = edge[:2]
            p_s = predicates[i]
            if np.array([none_p in p_s for none_p in ['and',  'unrelated', 'not', 'none']]).any():
                # if p_s in ['', 'and', ' ', 'unrelated', 'not', 'none']:
                continue
            dot.edge(str(s.item()), str(o.item()), p_s)
        dot.render(filename, format="png")

    def mRecall_objects(self, hits, misses):
        misses = np.concatenate(misses, axis=0).squeeze().tolist()
        hits = np.concatenate(hits, axis=0).squeeze().tolist()
        if type(misses) is not list:
            misses = [misses]
        if type(hits) is not list:
            hits = [hits]
        class_ids = list(range(len(self.obj_class_dict)))
        if type(hits) == int:
            hits = [hits]
        total_counts = {class_id: (hits+misses).count(class_id) for class_id in class_ids}
        hit_counts = {class_id: hits.count(class_id) for class_id in class_ids}
        miss_counts = {class_id: misses.count(class_id) for class_id in class_ids}
        percent_hits = {k: v/total_counts[k] for k, v in hit_counts.items() if total_counts[k] > 0}
        percent_hits = [percent_hits.get(key, np.nan) for key in class_ids]
        return dict(zip(class_ids, percent_hits))

    def mRecall_predicates(self, hits, misses):
        misses = np.concatenate(misses, axis=0).squeeze().tolist()
        hits = np.concatenate(hits, axis=0).squeeze().tolist()
        if type(misses) is not list:
            misses = [misses]
        if type(hits) is not list:
            hits = [hits]
        class_ids = list(range(len(self.pred_class_dict_orig)))
        total_counts = {class_id: (hits+misses).count(class_id) for class_id in class_ids}
        hit_counts = {class_id: hits.count(class_id) for class_id in class_ids}
        miss_counts = {class_id: misses.count(class_id) for class_id in class_ids}
        percent_hits = {k: v/total_counts[k] for k, v in hit_counts.items() if total_counts[k] > 0}
        percent_hits = [percent_hits.get(key, np.nan) for key in class_ids]
        return dict(zip(class_ids, percent_hits))

    def mRecall_relationships_by_pred(self, hits, misses):
        misses = np.concatenate(misses, axis=1).squeeze()[:, 1].squeeze().tolist()
        hits = np.concatenate(hits, axis=1).squeeze()[:, 1].squeeze().tolist()
        if type(misses) is not list:
            misses = [misses]
        if type(hits) is not list:
            hits = [hits]
        class_ids = self.pred_class_dict_orig  # list(range(len(self.pred_class_dict)))
        total_counts = {class_id: (hits+misses).count(class_id) for class_id in class_ids}
        hit_counts = {class_id: hits.count(class_id) for class_id in class_ids}
        miss_counts = {class_id: misses.count(class_id) for class_id in class_ids}
        percent_hits = {k: v/total_counts[k] for k, v in hit_counts.items() if total_counts[k] > 0}
        percent_hits = [percent_hits.get(key, np.nan) for key in class_ids]
        return dict(zip(class_ids, percent_hits))

    def mRecall_relationships(self, hits, misses):
        misses = np.concatenate(misses, axis=1).squeeze().squeeze().tolist()
        misses = ['_'.join(sublist) for sublist in misses]
        if type(misses) is not list:
            misses = [misses]
        if type(hits) is not list:
            hits = [hits]
        hits = np.concatenate(hits, axis=1).squeeze().squeeze().tolist()
        hits = ['_'.join(sublist) for sublist in hits]
        class_ids = np.unique(np.array(misses+hits)).tolist()
        total_counts = {class_id: (hits+misses).count(class_id) for class_id in class_ids}
        hit_counts = {class_id: hits.count(class_id) for class_id in class_ids}
        miss_counts = {class_id: misses.count(class_id) for class_id in class_ids}
        percent_hits = {k: v/total_counts[k] for k, v in hit_counts.items() if total_counts[k] > 0}
        percent_hits = [percent_hits.get(key, np.nan) for key in class_ids]
        return dict(zip(class_ids, percent_hits)), total_counts

    def dump_stats(self, hits, misses, name):
        misses = np.concatenate(misses, axis=0).squeeze().tolist()
        hits = np.concatenate(hits, axis=0).squeeze().tolist()
        class_ids = list(range(len(self.obj_class_dict)))
        total_counts = {class_id: (hits+misses).count(class_id) for class_id in class_ids}
        hit_counts = {class_id: hits.count(class_id) for class_id in class_ids}
        miss_counts = {class_id: misses.count(class_id) for class_id in class_ids}
        percent_hits = {k: v/total_counts[k] for k, v in hit_counts.items() if total_counts[k] > 0}
        percent_hits = [percent_hits.get(key, np.nan) for key in class_ids]
        dict(zip(class_ids, percent_hits))

        dump_dir = self.trainer.logger.log_dir
        dump_dir = dump_dir.replace('outputs', 'classwise_eval')
        os.makedirs(dump_dir, exist_ok=True)
        json.dump(dict(zip(class_ids, percent_hits)), open(os.path.join(dump_dir, f'{name}_eval_percent.json'), "w"))
        json.dump(dict(zip(class_ids, {k: hit_counts.get(k, 0) for k in class_ids})),
                  open(os.path.join(dump_dir, f'{name}_eval_hits.json'), "w"))
        json.dump(dict(zip(class_ids, {k: miss_counts.get(k, 0) for k in class_ids})),
                  open(os.path.join(dump_dir, f'{name}_eval_misses.json'), "w"))

    def dump_stats_rel(self, hits, misses, name):
        misses = np.concatenate(misses, axis=1).squeeze().tolist()
        hits = np.concatenate(hits, axis=1).squeeze().tolist()
        class_ids = list(range(len(self.pred_class_dict_orig)))
        total_counts = {class_id: (hits+misses).count(class_id) for class_id in class_ids}
        hit_counts = {class_id: hits.count(class_id) for class_id in class_ids}
        miss_counts = {class_id: misses.count(class_id) for class_id in class_ids}
        percent_hits = {k: v/total_counts[k] for k, v in hit_counts.items() if total_counts[k] > 0}
        percent_hits = [percent_hits.get(key, np.nan) for key in class_ids]
        dict(zip(class_ids, percent_hits))

        dump_dir = self.trainer.logger.log_dir
        dump_dir = dump_dir.replace('outputs', 'classwise_eval')
        os.makedirs(dump_dir, exist_ok=True)
        json.dump(dict(zip(class_ids, percent_hits)), open(os.path.join(dump_dir, f'{name}_eval_percent.json'), "w"))
        json.dump(dict(zip(class_ids, {k: hit_counts.get(k, 0) for k in class_ids})),
                  open(os.path.join(dump_dir, f'{name}_eval_hits.json'), "w"))
        json.dump(dict(zip(class_ids, {k: miss_counts.get(k, 0) for k in class_ids})),
                  open(os.path.join(dump_dir, f'{name}_eval_misses.json'), "w"))
