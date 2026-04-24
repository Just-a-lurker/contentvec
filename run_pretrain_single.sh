##!/bin/bash
#
#expdir=./tmp
#mkdir -p $expdir
#
#
#HYDRA_FULL_ERROR=1 python -u ./fairseq/fairseq_cli/hydra_train.py  \
#    --config-dir ./contentvec/config/contentvec \
#    --config-name contentvec \
#    hydra.run.dir=${expdir} \
#    task.data=./metadata \
#    task.label_dir=./label \
#    task.labels=["km"] \
#    task.spk2info=./spk2info.dict \
#    task.crop=true \
#    dataset.train_subset=train \
#    dataset.valid_subset=valid \
#    dataset.num_workers=10 \
#    dataset.max_tokens=500000 \
#    checkpoint.keep_best_checkpoints=10 \
#    criterion.loss_weights=[10,1e-5] \
#    model.label_rate=50 \
#    model.encoder_layers_1=3 \
#    model.logit_temp_ctr=0.1 \
#    model.ctr_layers=[-6] \
#    model.extractor_mode="default" \
#    optimization.update_freq=[1] \
#    optimization.max_update=100000 \
#    lr_scheduler.warmup_updates=8000

$DATA_ROOT = "D:/Tools/Projects/contentvec/dataset/manifest"
$EXP_DIR = "D:/Tools/Projects/contentvec/dataset/exp_finetune"
$PRETRAINED_CKPT = "D:/Tools/Projects/contentvec/dataset/exp_finetune/checkpoints/checkpoint_last.pt"

$env:HYDRA_FULL_ERROR = "1"
python -u ./fairseq/fairseq_cli/hydra_train.py `
    --config-dir ./contentvec/config/contentvec `
    --config-name contentvec `
    common.fp16=true `
    checkpoint.save_interval_updates=1000 `
    hydra.run.dir="$EXP_DIR" `
    task.data="$DATA_ROOT" `
    task.label_dir="$DATA_ROOT" `
    task.labels='["km"]' `
    task.spk2info="$DATA_ROOT/spk2info.dict" `
    task.crop=true `
    task.max_sample_size=80000 `
    dataset.train_subset=train `
    dataset.valid_subset=valid `
    dataset.num_workers=0 `
    dataset.max_tokens=80000 `
    checkpoint.restore_file="$PRETRAINED_CKPT" `
    checkpoint.keep_best_checkpoints=10 `
    criterion.loss_weights=[5,1e-4] `
    model.label_rate=50 `
    model.encoder_layers_1=3 `
    model.logit_temp_ctr=0.1 `
    model.ctr_layers=[-6] `
    model.extractor_mode="default" `
    optimization.update_freq=[3] `
    optimization.max_update=30000 `
    lr_scheduler.warmup_updates=400 `
    optimization.lr=[0.00005] `
    +model.freeze_encoder_layers=0 `
#    checkpoint.reset_optimizer=true `
#    checkpoint.reset_lr_scheduler=true `
#    checkpoint.reset_meters=true `
    #    model.feature_grad_mult=0.0 `





