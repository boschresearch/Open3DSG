<!-- PROJECT LOGO -->

<p align="center">
<h1>
  Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships
</h1>
  <p align="center">
    <a href="https://kochsebastian.com/"><strong>Sebastian Koch</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=U3KSTwkAAAAJ&hl=en"><strong>Narunas Vaskevicius</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=k4m1c6EAAAAJ"><strong>Mirco Colosi</strong></a>
    <br>
    <a href="https://phermosilla.github.io/"><strong>Pedro Hermosilla</strong></a>
    ·
    <a href="https://viscom.uni-ulm.de/members/timo-ropinski/"><strong>Timo Ropinski</strong></a>
  </p>
  <h2 align="center">CVPR 2024</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2402.12259">Paper</a> | <a href="https://kochsebastian.com/open3dsg">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/kochsebastian/kochsebastian.github.io/blob/master/media/open3dsg/teaser.png?raw=true" alt="Logo" width="85%">
  </a>
</p>
<p align="center">
We present <strong>Open3DSG</strong> the first approach
for learning to predict open-vocabulary 3D scene graphs from
3D point clouds. The advantage of our method is that it can be
queried and prompted for any instance in the scene, such as the
TV and Wall, to predict fine-grained semantic descriptions of objects and relationships.
</p>
<br>

## Setup

```bash
conda env create --name open3dsg python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

> **Note**: This software is tested and developed for CUDA 11.8 & tested with an NVIDIA V100 32GB.

### Data Preparation

1. Download [3RScan](https://github.com/WaldJohannaU/3RScan) and [3DSSG](https://3dssg.github.io/). Unpack the image sequences for each scan. And include the 3DSSG files as a subdirectory in 3RScan.
2. Download [ScanNet](http://www.scan-net.org/ScanNet/) and split the scans into ```scannet_2d``` and ```scannet_3d```. We use the pre-processed data from [ScanNet ETH preprocessed 3D](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip) & [ScanNet ETH preprocessed 2D](https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip), when using the pre-processed version make sure that you have acknowledged the ScanNet license. When using processed ScanNet ETH preprocessed 2D frames, use the matching [intrinsics](https://drive.google.com/drive/folders/1rlzUS1d5cYo5lJCNl1G81x9HmYtn5NB5?usp=drive_link).
3. Download the [3DSSG_subset.zip](http://campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset.zip) and extract the files in the 3RScan directory for training and evaluation. Additional meta files can be found [here](https://drive.google.com/drive/folders/1rlzUS1d5cYo5lJCNl1G81x9HmYtn5NB5?usp=drive_link).
4. Download 3RScan & ScanNet meta data files using ```scripts/download_scannet_meta.sh``` and ```scripts/download_scannet_meta.sh``` and place them in their data directories.
5. Set the path to your data in ```config/config.py```

### Data Preprocessing

3DSSG provides pre-constructed scene graphs with ground-truth labels for training and validation. ScanNet does not. To train our model on ScanNet, we first have to build up a similar graph structure for ScanNet. You can use the following command to generate the graphs for ScanNet

```bash
python open3dsg/data/gen_scannet_subgraphs.py --type [train/test/validation]
```

For the 2D-3D distillation training, we have to align the 2D frames to the 3D point clouds. Using this script we generate matching frames for each 3D instance.

```bash
python open3dsg/data/get_object_frame.py --mode [train/test] --dataset [R3SCAN/SCANNET]
```

We pre-process the data before the training for faster data processing in the training loop.

```bash
python open3dsg/data/preprocess_3rscan.py
python open3dsg/data/preprocess_scannet.py
```

The pre-processed features can be used directly for training and testing.

### Model Downloads

Download the [OpenSeg Checkpoint](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/openseg), [BLIP2 Positional Embedding](https://drive.google.com/file/d/1BfvxB6eo3XksE6AfMUgoBHwzVYce1ed1/view?usp=sharing) & pre-trained [PointNet/PointNet2 weights](https://drive.google.com/drive/folders/1PrnJVMpJVVh4MAV4yPRuRByhBu-DuXwH?usp=sharing) and put them the checkpoints directory selected in the config file.

## Precompute 2D features

This is an **optional** step to accelerate the forward pass in the training loop. This command will dump the VLM features for each training sample to disk. Storing the features requires about 300GB per dataset.

```bash
python open3dsg/scripts/run.py --dump_features --dataset [scannet/3rscan] --scales 3 --top_k_frames 5 --clip_model OpenSeg --blip
```

In case of out of memory issues, seperate the BLIP export & the OpenSeg export.

## Train

To train Open3DSG on ScanNet you can use:

```bash
python open3dsg/scripts/run.py --epochs 100 --batch_size 4 --gpus 4 --workers 8 --use_rgb --dataset scannet --clip_model OpenSeg --blip --load_features [path to precomputed 2D features]
```

Change hyperparameters according to you hardware availability. In [run.py](open3dsg/scripts/run.py) you can find more model and data hyperparameters.
Use ```--mixed_precision``` to optimize GPU memory during training.

## Test

To evaluate a trained model on the 3RSCAN dataset with ground-truth labels, use the following command:

```bash
python open3dsg/script/run.py --test --dataset 3rscan --checkpoint [path to checkpoint] --n_beams 5 --weight_2d 0.5 --clip_model OpenSeg --node_model ViT-L/14@336px --blip
```

We use the ```CLIP ViT-L/14@336px``` to query object classes from the node embedding. Use ```--n_beams``` to adjust the beam search for the LLM relationship output and ```--weight_2d``` to adjust the 2D-3D features fusion. A value of 0.0 indicates a prediction from 3D features only

## Citation

If you find our code or paper useful, please cite

```bibtex
@inproceedings{koch2024open3dsg,
      title={Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships},
      author={Koch, Sebastian  and Vaskevicius, Narunas and Colosi, Mirco and Hermosilla, Pedro and Ropinski, Timo},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month={June},
      year={2024},
  }
```

## License

Open3DSG is open-sourced under the AGPL-3.0 license. See the LICENSE file for details.

For a list of other open source components included in Open3DSG, see the file 3rd-party-licenses.txt.
