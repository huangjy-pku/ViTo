## Vision as Tokens

#### TODO

1. utils.vis_mask, visualize masks
2. consider offline augmentation w.r.t. dense task

#### Installation

- Environment

  ```bash
  conda create -n vito python=3.8
  conda activate vito
  ```

- Requirements

  ```bash
  pip install -r requirements.txt
  ```

- Compile CUDA

  ```bash
  cd ./models/ops
  sh ./make.sh
  python test.py   # unit test (should see all checking is True)
  ```

#### Data preparation

- Refer to [lichengunc/refer: Referring Expression Datasets API (github.com)](https://github.com/lichengunc/refer), prepare images and json files in order. Images consist of MS COCO Train2014 and Saiapr_tc-12.
- Configure data/generate_anno.py and run it to generate annotations and masks on four datasets (RefClef, RefCoco, RefCoco+, RefCocog).