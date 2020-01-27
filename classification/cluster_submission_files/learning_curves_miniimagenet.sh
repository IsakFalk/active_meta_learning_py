#$ -S /bin/bash
#$ -j y
#$ -N aml_omniglot
#$ -t 1-10
#$ -wd /cluster/project9/MMD_FW_active_meta_learning

#$ -l tmem=8G
#$ -l h_rt=8:0:0
#$ -l gpu=true

PROJECT_DIR=/cluster/project9/MMD_FW_active_meta_learning
DATA_DIR=$PROJECT_DIR/data/
SAVE_DIR=$PROJECT_DIR/experiments/learning_curves/${JOB_ID}_cnn_miniimagenet_kh
mkdir $SAVE_DIR

N_TRAIN_BATCHES=10000
N_TEST_BATCHES=200
TASKS_PER_METAUPDATE=16
EVALUATE_EVERY=50

N_WAY=5
K_SHOT=1
K_QUERY=15

LR_INNER=0.4
LR_META=0.001

FRANK_WOLFE=kernel_herding
KERNEL_FUNCTION=mean_linear

NUM_GRAD_STEPS_INNER=1
NUM_GRAD_STEPS_EVAL=1

NUM_FILTERS=32

DATASET=omniglot
BASE_DATASET_TRAIN=train
BASE_DATASET_VAL=val
BASE_DATASET_TEST=test
DATA_PATH=$DATA_DIR
SAVE_PATH=$SAVE_DIR
N_WORKERS=0

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
        --n_train_batches $N_TRAIN_BATCHES \
        --n_test_batches $N_TEST_BATCHES \
        --seed $SGE_TASK_ID \
        --tasks_per_metaupdate $TASKS_PER_METAUPDATE \
        --evaluate_every $EVALUATE_EVERY \
p        --n_way $N_WAY \
        --k_shot $K_SHOT \
        --k_query $K_QUERY \
        --lr_inner $LR_INNER \
        --lr_meta $LR_META \
        --frank_wolfe $FRANK_WOLFE \
        --kernel_function $KERNEL_FUNCTION \
        --num_grad_steps_inner $NUM_GRAD_STEPS_INNER \
        --num_grad_steps_eval $NUM_GRAD_STEPS_EVAL \
        --first_order $FIRST_ORDER \
        --model $MODEL \
        --num_filters $NUM_FILTERS \
        --hidden_size $HIDDEN_SIZE \
        --num_layers $NUM_LAYERS \
        --dataset $DATASET \
        --base_dataset_train $BASE_DATASET_TRAIN \
        --base_dataset_val $BASE_DATASET_VAL \
        --base_dataset_test $BASE_DATASET_TEST \
        --data_path $DATA_PATH \
        --save_path $SAVE_PATH \
        --n_workers $N_WORKERS \
        --write_config $WRITE_CONFIG

date
