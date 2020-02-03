#$ -S /bin/bash
#$ -j y
#$ -N aml_sine_maml
#$ -t 1-5
#$ -wd /cluster/project9/MMD_FW_active_meta_learning/active_meta_learning_py/regression_maml

#$ -l tmem=12G
#$ -l gpu=true
#$ -l h_rt=10:0:0

PROJECT_DIR=/cluster/project9/MMD_FW_active_meta_learning
SAVE_DIR=$PROJECT_DIR/experiments/learning_curves/${JOB_ID}_mlp_sine_kh_maml
mkdir -p $SAVE_DIR

# Remember to also export the path to the libraries and software needed, example
source $PROJECT_DIR/project_environment.source

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
        --n_train_batches 2000 \
        --n_test_batches 100 \
        --seed $SGE_TASK_ID \
        --tasks_per_metaupdate 1 \
        --evaluate_every 50 \
        --k_shot 5 \
        --k_query 15 \
        --lr_inner 0.5 \
        --lr_meta 0.2 \
        --meta_optimizer sgd \
        --frank_wolfe kernel_herding \
        --kernel_function double_gaussian_kernel \
        --num_grad_steps_inner 1 \
        --num_grad_steps_eval 1 \
        --num_grad_steps_meta 5 \
        --hidden_dim 40 \
        --dataset sine \
        --save_path $SAVE_DIR \
        --n_workers 0 \
        --write_config $WRITE_CONFIG

date
