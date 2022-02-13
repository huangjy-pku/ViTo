## Vision as Tokens

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

- Compile CUDA (in case of deformable mode)

  ```bash
  cd ./model/ops
  sh ./make.sh
  python test.py   # unit test (should see all checking is True)
  ```

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
  python train_distr.py exp_name=${your_exp_name} task=[dense]   # dense task
  python train_distr.py exp_name=${your_exp_name} task=[bbox,dense]   # joint training
  ```

#### Evaluation

- Evaluation on validation and test datasets

  ```bash
  python eval_testset.py exp_name=${your_exp_name} task={your_task}
  ```

  