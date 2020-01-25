# Use this to test / dry run your jobs
# Essentially running the code you would do
# with `qsub` by running `qrsh -t` with
# whatever -t options you have defined there
# Remove any scheduler directive lines,
# the ones starting with `#$`.

# Dummy variables, these will be set by the scheduler
# when running `qsub`
JOB_ID=TEST
SGE_TASK_ID=1

PROJECT_DIR=/cluster/project9/MMD_FW_active_meta_learning
DATA_DIR=$PROJECT_DIR/data/
SAVE_DIR=$PROJECT_DIR/experiments/learning_curves/${JOB_ID}_cnn_omniglot_kh
mkdir $SAVE_DIR

N_TRAIN_BATCHES=1000
N_TEST_BATCHES=100
TASKS_PER_METAUPDATE=16
EVALUATE_EVERY=10

N_WAY=5
K_SHOT=1
K_QUERY=15

LR_INNER=0.5
LR_META=0.001

FRANK_WOLFE=kernel_herding
KERNEL_FUNCTION=mean_linear

NUM_GRAD_STEPS_INNER=1
NUM_GRAD_STEPS_EVAL=1

MODEL=cnn
NUM_FILTERS=32
HIDDEN_SIZE=64
NUM_LAYERS=3

DATASET=omniglot
BASE_DATASET_TRAIN=train
BASE_DATASET_VAL=val
BASE_DATASET_TEST=test
DATA_PATH=$DATA_DIR
SAVE_PATH=$SAVE_DIR
N_WORKERS=0

# Remember to also export the path to the libraries and software needed, example
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH

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
        --n_way $N_WAY \
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
        --write_config $WRITE_CONFIG \

date
