# Copyright (c) 2024 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import numpy as np
import os

from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager
import json

from open3dsg.config.config import CONF

# OpenScene Material Subset
# query_materials = ['wooden', 'padded', 'glass','metal','ceramic','cardboard','plastic','carpet','stone','concrete']
# materials_gt = json.load(open(CONF.PATH.R3SCAN_RAW +'/materials_subset.json','r'))

class_names = np.array([l.rstrip() for l in open(os.path.join(CONF.PATH.R3SCAN_RAW, "classes.txt"), 'r')])
relationships_names = np.array([l.rstrip() for l in open(os.path.join(CONF.PATH.R3SCAN_RAW, "relationships.txt"), 'r')])
# relationships_custom_names = np.array([l.rstrip() for l in open(os.path.join(CONF.PATH.R3SCAN_RAW, "relationships_custom.txt"), 'r')])
# manual_rel_mapping = [l.rstrip().split(', ')[1:] for l in open(os.path.join(CONF.PATH.R3SCAN_RAW, 'manual_rel_mapping.txt'),'r')]
# manual_rel_lookup = dict(zip(relationships_names,manual_rel_mapping))

# rel2idx = dict(zip(relationships_names,range(len(relationships_names))))
# known_mapping = dict(zip(relationships_custom_names,relationships_custom_names))
# known_mapping['to the left of'] = 'left of'
# known_mapping['to the right of'] = 'right of'
# known_mapping['next to'] = 'close by' # 'none'
# known_mapping['above'] = 'higher than' # 'none'
# known_mapping['under'] = 'lower than'

# def map_rel2idx(class_name):
#     return rel2idx.get(class_name, -1)
# rel2idx_mapping = np.vectorize(map_rel2idx)
# rel2rel_mapping = np.vectorize(known_mapping.get)
# rel_manual_mapping_vec = np.vectorize(manual_rel_lookup.get)


def get_eval(data_dict, eval_relationship=True):

    batch_size = data_dict["objects_id"].size(0)
    eval_dict = {}
    eval_dict["top1_recall_o"], eval_dict["top5_recall_o"], eval_dict["top10_recall_o"] = [], [], []
    eval_dict["top1_recall_p"], eval_dict["top3_recall_p"], eval_dict["top5_recall_p"] = [], [], []
    eval_dict["top1_recall_rel"], eval_dict["top50_recall_rel"], eval_dict["top100_recall_rel"] = [], [], []

    eval_dict['top_1_missed_objects'], eval_dict['top_1_hit_objects'] = [], []
    eval_dict['top_5_missed_objects'], eval_dict['top_5_hit_objects'] = [], []
    eval_dict['top_10_missed_objects'], eval_dict['top_10_hit_objects'] = [], []

    eval_dict['top_1_missed_predicates'], eval_dict['top_1_hit_predicates'] = [], []
    eval_dict['top_3_missed_predicates'], eval_dict['top_3_hit_predicates'] = [], []
    eval_dict['top_5_missed_predicates'], eval_dict['top_5_hit_predicates'] = [], []

    eval_dict['top1_missed_relationships'], eval_dict['top1_hit_relationships'] = [], []
    eval_dict['top50_missed_relationships'], eval_dict['top50_hit_relationships'] = [], []
    eval_dict['top100_missed_relationships'], eval_dict['top100_hit_relationships'] = [], []

    for bidx in range(batch_size):
        object_count = data_dict["objects_count"][bidx].item()
        rel_count = data_dict["predicate_count"][bidx].item()
        object_pred = data_dict["objects_predict"][bidx][:object_count].cpu().numpy()
        object_probs = data_dict["objects_probs"][bidx][:object_count].cpu().numpy()
        object_cat = data_dict["objects_cat"][bidx][:object_count].cpu().numpy()
        predicate_count = data_dict["predicate_count"][bidx].item()
        predicate_pred = np.array(data_dict["predicates_mapped"][bidx][:predicate_count])
        predicate_pred_probs = np.array(data_dict["predicates_mapped_probs"][bidx][:predicate_count].cpu().float().numpy())
        predicate_dist = data_dict["predicate_min_dist"][bidx][:predicate_count]

        pairs = data_dict["pairs"][bidx][:predicate_count].cpu().numpy()
        edges = data_dict["edges"][bidx][:predicate_count].cpu().numpy()
        triples = data_dict["triples"][bidx].cpu().numpy()

        # filter out all 0 rows
        zero_rows = np.zeros(triples.shape).astype(np.uint8)
        mask = (triples == zero_rows)[:, :2]
        mask = ~ (mask[:, 0] & mask[:, 1])
        triples = triples[mask]

        predicate_cat = triples[:, 2]

        ignore_nones = False

        top_k_obj = np.array(topk_object(object_pred, object_cat, 20))
        eval_dict["top1_recall_o"].append(np.sum(top_k_obj <= 1)/len(top_k_obj))
        eval_dict["top5_recall_o"].append(np.sum(top_k_obj <= 5)/len(top_k_obj))
        eval_dict["top10_recall_o"].append(np.sum(top_k_obj <= 10)/len(top_k_obj))

        eval_dict['top_1_hit_objects'].append(object_cat[top_k_obj <= 1])
        eval_dict['top_1_missed_objects'].append(object_cat[top_k_obj > 1])
        eval_dict['top_5_hit_objects'].append(object_cat[top_k_obj <= 5])
        eval_dict['top_5_missed_objects'].append(object_cat[top_k_obj > 5])
        eval_dict['top_10_hit_objects'].append(object_cat[top_k_obj <= 10])
        eval_dict['top_10_missed_objects'].append(object_cat[top_k_obj > 10])

        top_k_preds = topk_ratio_pred(predicate_pred_probs, data_dict['predicate_edges']
                                      [bidx][:rel_count], predicate_dist, k=5, ignore_nones=ignore_nones)
        eval_dict["top1_recall_p"].append(np.sum((top_k_preds <= 1))/len(top_k_preds))
        eval_dict["top3_recall_p"].append(np.sum((top_k_preds <= 3))/len(top_k_preds))
        eval_dict["top5_recall_p"].append(np.sum((top_k_preds <= 5))/len(top_k_preds))

        if ignore_nones:
            predicate_cat_no_nones = predicate_cat[predicate_cat != 0]

            eval_dict['top_1_hit_predicates'].append(predicate_cat_no_nones[top_k_preds <= 1])
            eval_dict['top_1_missed_predicates'].append(predicate_cat_no_nones[top_k_preds > 1])
            eval_dict['top_3_hit_predicates'].append(predicate_cat_no_nones[top_k_preds <= 3])
            eval_dict['top_3_missed_predicates'].append(predicate_cat_no_nones[top_k_preds > 3])
            eval_dict['top_5_hit_predicates'].append(predicate_cat_no_nones[top_k_preds <= 5])
            eval_dict['top_5_missed_predicates'].append(predicate_cat_no_nones[top_k_preds > 5])
        else:
            eval_dict['top_1_hit_predicates'].append(predicate_cat[top_k_preds <= 1])
            eval_dict['top_1_missed_predicates'].append(predicate_cat[top_k_preds > 1])
            eval_dict['top_3_hit_predicates'].append(predicate_cat[top_k_preds <= 3])
            eval_dict['top_3_missed_predicates'].append(predicate_cat[top_k_preds > 3])
            eval_dict['top_5_hit_predicates'].append(predicate_cat[top_k_preds <= 5])
            eval_dict['top_5_missed_predicates'].append(predicate_cat[top_k_preds > 5])

        eval_relationship = True
        if eval_relationship:

            top_k_rels = np.array(topk_relationship(object_probs, object_cat, predicate_pred_probs, predicate_dist,
                                  data_dict['predicate_edges'][bidx], edges, triples, k=100, ignore_nones=ignore_nones))
            eval_dict["top1_recall_rel"].append(np.sum((top_k_rels) <= 1)/len(top_k_rels))
            # eval_dict["top10_recall_rel"].append(np.sum((top_k_rels)<=10)/len(top_k_rels))
            eval_dict["top50_recall_rel"].append(np.sum((top_k_rels) <= 50)/len(top_k_rels))
            eval_dict["top100_recall_rel"].append(np.sum((top_k_rels) <= 100)/len(top_k_rels))

            # predicate_edges_flat = [item for sublist in data_dict['predicate_edges'][bidx] for item in sublist]
            # aligned_triples = []
            # for m in range(len(data_dict['edge_mapping'][bidx])-1):
            #     preds = np.array(predicate_edges_flat[data_dict['edge_mapping'][bidx][m]:data_dict['edge_mapping'][bidx][m+1]])[:,None]
            #     objs = np.repeat(object_cat[edges[m][None]], len(preds), axis=0)
            #     objs = class_names[objs]
            #     preds = relationships_names[preds]
            #     aligned_triples.append(np.stack(np.stack((objs[:,0],preds[:,0],objs[:,1]),axis=-1),axis=0))

            # eval_dict['top1_missed_relationships'].append(np.concatenate(aligned_triples,axis=0)[[(top_1_rels)>1]]); eval_dict['top1_hit_relationships'].append(np.concatenate(aligned_triples,axis=0)[[(top_1_rels)<=1]])
            # eval_dict['top50_missed_relationships'].append(np.concatenate(aligned_triples,axis=0)[[(top_k_rels)>50]]); eval_dict['top50_hit_relationships'].append(np.concatenate(aligned_triples,axis=0)[[(top_k_rels)<=50]])
            # eval_dict['top100_missed_relationships'].append(np.concatenate(aligned_triples,axis=0)[[(top_k_rels)>100]]); eval_dict['top100_git_relationships'].append(np.concatenate(aligned_triples,axis=0)[[(top_k_rels)<=100]])
        else:
            eval_dict["top1_recall_rel"].append(0)
            eval_dict["top50_recall_rel"].append(0)
            eval_dict["top100_recall_rel"].append(0)

    data_dict.update(eval_dict)
    return data_dict


def topk_object(objs_pred, objs_target, k=-1):
    top_k = list()
    size_o = len(objs_pred)

    sorted_args = objs_pred[:, :k]
    for obj in range(size_o):
        obj_pred = sorted_args[obj]
        gt = objs_target[obj]
        indices = np.where(obj_pred == gt)[0]
        if len(indices) == 0:
            index = np.inf
        else:
            index = sorted(indices)[0].item()+1
        top_k.append(index)

    return top_k


def topk_ratio_pred(logits, category, dist, k, ignore_nones=False):
    top_k = []

    sorted_conf, sorted_args = torch.sort(torch.from_numpy(logits[:, :]), dim=1, descending=True)
    sorted_conf = sorted_conf[:, :k]
    sorted_args = sorted_args[:, :k]

    for i, (confs, args) in enumerate(zip(sorted_conf, sorted_args)):
        for c in category[i]:
            if c == 0 and ignore_nones:
                continue
                indices = torch.where(confs < 0.5)[0]
                if len(indices) == 0:
                    index = np.inf
                else:
                    index = sorted(indices)[0].item()+1
                top_k.append(index)
            else:
                if c == 0:
                    if np.linalg.norm(dist[i]) > 0.5:
                        index = 1
                    else:
                        indices = torch.where(confs == logits[i][c])[0]
                        if len(indices) == 0:
                            index = np.inf
                        else:
                            index = sorted(indices)[0].item()+1
                else:
                    indices = torch.where(confs == logits[i][c])[0]
                    if len(indices) == 0:
                        index = np.inf
                    else:
                        index = sorted(indices)[0].item()+1
                top_k.append(index)

    return np.array(top_k)


def topk_relationship(object_pred, object_cat, predicate_pred, predicate_dist, predicate_edges, edges, triples, k=50, ignore_nones=False):
    object_seg_edges = object_cat[edges]
    top_k = list()
    dists_05 = []

    for j, edge in enumerate(edges):
        objs_pred_1 = object_pred[edge[0]]
        objs_pred_2 = object_pred[edge[1]]
        rel_predictions = predicate_pred[j][:]  # ignore None predictions

        node_score = np.einsum('n,m->nm', objs_pred_1, objs_pred_2)
        conf_matrix = np.einsum('nl,m->nlm', node_score, rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_conf_matrix, sorted_args_1d = torch.sort(torch.from_numpy(conf_matrix_1d), descending=True)
        sorted_conf_matrix = sorted_conf_matrix[:k]
        sorted_args_1d = sorted_args_1d[:k]

        predicate_edges[j]
        gt_s = object_seg_edges[j][0]
        gt_t = object_seg_edges[j][1]
        gt_r = predicate_edges[j]
        temp_topk = []
        for predicate in gt_r:
            dists_05.append(np.linalg.norm(predicate_dist[j]) <= 0.5)
            if predicate == 0 and ignore_nones:
                continue
            elif predicate == 0:  # and ignore_nones:
                gt_confs = conf_matrix[:, :, predicate]
                if len(list(set(sorted_conf_matrix.numpy()).intersection(gt_confs.reshape(-1)))) > 0:
                    index = [t for t, s in enumerate(sorted_conf_matrix) if s.item() in gt_confs][0]+1
                else:
                    index = k+1
                # indices = torch.where(sorted_conf_matrix < 0.5)[0]
                # if len(indices) == 0:
                #     index = k+1
                # else:
                #     index = sorted(indices)[0].item()+1
            else:
                gt_conf = conf_matrix[int(gt_s), int(gt_t), predicate]
                indices = torch.where(sorted_conf_matrix == gt_conf)[0]
                if len(indices) == 0:
                    index = np.inf
                else:
                    index = sorted(indices)[0].item()+1
            temp_topk.append(index)
        temp_topk = sorted(temp_topk)
        top_k += temp_topk
    return top_k


def eval_attribute(data_dict):
    recall_per_class = [[] for _ in range(len(query_materials))]
    data_dict['topk_graph'] = []

    def material_topk(m_pred, materials_gt, object_ids, scan_id, k=1):
        recall = [[] for _ in range(len(query_materials))]
        for obj_id, m in zip(object_ids, m_pred):
            m_gt = materials_gt['-'.join(scan_id.split('-')[:-1])].get(str(int(obj_id.item())), [])
            if len(m_gt) == 0:
                continue
            for gt in m_gt:
                indices = np.where(np.array(query_materials)[m] == gt)
                if len(indices) == 0:
                    index = np.inf
                else:
                    index = sorted(indices)[0].item()+1

                recall[query_materials.index(gt)].append(index)
        return recall

    batch_size = data_dict["objects_id"].size(0)
    for bidx in range(batch_size):
        object_count = data_dict["objects_count"][bidx].item()
        objects_id = data_dict["objects_id"][bidx][:object_count]
        material_pred = data_dict["materials_predict"][bidx][:object_count]

        scan_id = data_dict["scan_id"][bidx]

        recall_att_graph = material_topk(material_pred, materials_gt, objects_id, scan_id)
        for i, r in enumerate(recall_att_graph):
            recall_per_class[i].extend(r)
        data_dict['topk_graph'].append(recall_per_class)
    return data_dict
