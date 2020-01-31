#$ -S /bin/bash
#$ -j y
#$ -N aml_sine
#$ -t 1-5
#$ -wd /cluster/project9/MMD_FW_active_meta_learning/active_meta_learning_py/regression

#$ -l tmem=20G
#$ -l h_rt=10:0:0

SGE_TASK_ID=1
JOB_ID=TEST
PROJECT_DIR=/home/isak/life/references/projects/phd/active_meta_learning/src/active_meta_learning_py/active_meta_learning/regression
#PROJECT_DIR=/cluster/project9/MMD_FW_active_meta_learning
SAVE_DIR=$PROJECT_DIR/experiments/learning_curves/${JOB_ID}_mlp_sine_kh
mkdir -p $SAVE_DIR

N_TRAIN_BATCHES=100
N_TEST_BATCHES=10
TASKS_PER_METAUPDATE=1
EVALUATE_EVERY=50

K_SHOT=5
K_QUERY=15

LR_INNER=0.01
LR_META=0.001

FRANK_WOLFE=kernel_herding
KERNEL_FUNCTION=double_gaussian_kernel

NUM_GRAD_STEPS_INNER=1
NUM_GRAD_STEPS_EVAL=1

HIDDEN_DIM=40
DATASET=sine

SAVE_PATH=$SAVE_DIR
N_WORKERS=0

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
        --n_train_batches $N_TRAIN_BATCHES \
        --n_test_batches $N_TEST_BATCHES \
        --seed $SGE_TASK_ID \
        --tasks_per_metaupdate $TASKS_PER_METAUPDATE \
        --evaluate_every $EVALUATE_EVERY \
        --k_shot $K_SHOT \
        --k_query $K_QUERY \
        --lr_inner $LR_INNER \
        --lr_meta $LR_META \
        --frank_wolfe $FRANK_WOLFE \
        --kernel_function $KERNEL_FUNCTION \
        --num_grad_steps_inner $NUM_GRAD_STEPS_INNER \
        --num_grad_steps_eval $NUM_GRAD_STEPS_EVAL \
        --hidden_dim $HIDDEN_DIM \
        --dataset $DATASET \
        --save_path $SAVE_PATH \
        --n_workers $N_WORKERS \
        --write_config $WRITE_CONFIG

date
