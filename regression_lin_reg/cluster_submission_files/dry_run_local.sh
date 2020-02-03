#$ -S /bin/bash
#$ -j y
#$ -N aml_lin_reg_sine
#$ -t 1-5
#$ -wd /cluster/project9/MMD_FW_active_meta_learning/active_meta_learning_py/regression_lin_reg

#$ -l tmem=20G
#$ -l h_rt=10:0:0

SGE_TASK_ID=1
JOB_ID=TEST
PROJECT_DIR=/home/isak/life/references/projects/phd/active_meta_learning/src/active_meta_learning_py/active_meta_learning/regression_lin_reg
#PROJECT_DIR=/cluster/project9/MMD_FW_active_meta_learning
SAVE_DIR=$PROJECT_DIR/experiments/learning_curves/${JOB_ID}_mlp_sine_kh
mkdir -p $SAVE_DIR
SAVE_PATH=$SAVE_DIR

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
        --n_train_batches 300\
        --n_val_batches 1 \
        --n_test_batches 10 \
        --seed $SGE_TASK_ID \
        --tasks_per_metaupdate 4 \
        --evaluate_every 10 \
        --k_shot 5 \
        --k_query 15 \
        --inner_regularization 0.05 \
        --lr_meta 0.4 \
        --meta_optimizer sgd \
        --frank_wolfe kernel_herding \
        --kernel_function double_gaussian_kernel \
        --num_grad_steps_meta 1 \
        --hidden_dim 40 \
        --feature_dim 10 \
        --dataset sine \
        --save_path $SAVE_PATH \
        --write_config $WRITE_CONFIG

date
