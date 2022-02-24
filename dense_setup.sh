# aug_offline first to generate samples for dense training
EXP_NAME=$1
python aug_offline.py
python train_distr.py exp_name=${EXP_NAME} task=[dense]