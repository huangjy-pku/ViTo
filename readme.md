TODO:
    1. model.vito.ViTo.load_pretr_weights, to align with module names of pretrained checkpoints
    2. model.vito.ViTo.vocab_expansion, expand codebook of dense tasks
    3. consider offline augmentation w.r.t. dense task
    4. dataset.generic_dataset.dense_process, generate ground truth sequences by VQGAN encoder
    5. utils.seq2mask and utils.vis_mask, process sequences and visualize masks