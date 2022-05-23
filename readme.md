## Vision as Tokens

- This repo is the implementation of my bachelor thesis: *Towards Unifying Visual Discriminative Tasks from Generative Representation*, which is not published yet.

- We provide two settings, named MDETR and VQGAN respectively, to uniformly handle three visual discriminative tasks:
  - Phrase Grounding
  - Phrase Segmentation
  - Depth Estimation
- Note: this repo may be not well-organized, and if you have any confusion or relevant idea, feel free to contact me at huangjiangyong@pku.edu.cn.

#### Installation

- Environment

  ```bash
  conda create -n vito python=3.8
  conda activate vito
  ```

- Requirements

  ```bash
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  pip install -r requirements.txt
  ```

- Dependencies about VQGAN. Please refer to '*Training on custom data*' in [VQGAN](https://github.com/CompVis/taming-transformers) for setup. Specify the configurations and finish stage-one training, which we are going to utilize during stage-two for quantized representation.


#### Data preparation

- Refer to [lichengunc/refer: Referring Expression Datasets API (github.com)](https://github.com/lichengunc/refer), prepare images and json files in order. Images consist of MS COCO Train2014 and Saiapr_tc-12.
- Configure data/generate_anno.py and run it to generate annotations and masks on four datasets (RefClef, RefCoco, RefCoco+, RefCocog).

#### Configuration

- Modify config/vito.yaml
  - output_dir, holding running results
  - refexp_data, which contains a folder and two subfolders: images/mscoco, images/saiapr_tc-12
  - refexp_anno, partitioned into subfolders of datasets, each of which containing json annotataions
  - refexp_mask, partitioned into subfolders of datasets, each of which containing binary masks

#### Training

- Run distributed training

  ```bash
    python train_distr.py exp_name=${your_exp_name} task=[bbox]   # bbox task
    
    python train_distr.py exp_name=${your_exp_name} task=[mask]   # mask task
    
    python train_distr.py exp_name=${your_exp_name} task=[depth]   # depth task
    
    # joint training
    python train_distr.py exp_name=${your_exp_name} task=[bbox,mask,depth]
  ```

#### Evaluation

- Evaluation on validation and test datasets

  ```bash
  python eval_testset.py exp_name=${your_exp_name} task={your_task}
  ```
