#$ -S /bin/bash
#$ -j y
#$ -N aml_omniglot_maml
# #$ -t 1-5
#$ -wd /cluster/project9/MMD_FW_active_meta_learning/active_meta_learning_py/classification

#$ -l tmem=2G
#$ -l h_rt=10:0:0
#$ -l gpu=true

SGE_TASK_ID=1
JOB_ID=TEST
PROJECT_DIR=/home/isak/life/references/projects/phd/active_meta_learning/src/active_meta_learning_py/active_meta_learning/classification
#PROJECT_DIR=/cluster/project9/MMD_FW_active_meta_learning
DATA_DIR=$PROJECT_DIR/data/
SAVE_DIR=$PROJECT_DIR/experiments/learning_curves/${JOB_ID}_cnn_omniglot_kh
mkdir -p $SAVE_DIR

# Remember to also export the path to the libraries and software needed, example
#source $PROJECT_DIR/project_environment.source

# Log host and date
hostname
date

# Only write config once
if [ "$SGE_TASK_ID" -eq "1" ]
then
    WRITE_CONFIG=yes
else
    WRITE_CONFIG=no
fi

# Run it with seed specified by task id
# NOTE: If you want to specify first order,
# add --first_order flag to end.
python3 main.py \
        --n_train_batches 10 \
        --n_test_batches 2 \
        --seed $SGE_TASK_ID \
        --tasks_per_metaupdate 2 \
        --evaluate_every 1 \
        --k_shot 1 \
        --k_query 5 \
        --lr_inner 0.5 \
        --lr_meta 0.4 \
        --meta_optimizer sgd \
        --frank_wolfe kernel_herding \
        --kernel_function mean_linear \
        --num_grad_steps_inner 1 \
        --num_grad_steps_eval 1 \
        --num_grad_steps_meta 1 \
        --num_filters 64 \
        --dataset omniglot \
        --base_dataset_train train \
        --base_dataset_val val \
        --base_dataset_test test \
        --data_path $DATA_DIR \
        --save_path $SAVE_DIR \
        --n_workers 0 \
        --write_config $WRITE_CONFIG

date
