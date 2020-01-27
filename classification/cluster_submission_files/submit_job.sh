#$ -S /bin/bash
#$ -j y
#$ -N aml_omniglot

#$ -l tmem=8G
#$ -l h_rt=8:0:0
#$ -l gpu=true
#$ -wd /home/jitfalk/stdouterr/

PROJECT_DIR=/cluster/project9/MMD_FW_active_meta_learning
DATA_DIR=$PROJECT_DIR/data/
SAVE_DIR=$PROJECT_DIR/experiments/learning_curves

N_TRAIN_BATCHES=5000
n_test_batches=200
tasks_per_metaupdate=16
evaluate_every=50

n_way=5
k_shot=1
k_query=15

lr_inner=0.5
lr_meta=0.001

frank_wolfe=kernel_herding
kernel_function=mean_linear

num_grad_steps_inner=1
num_grad_steps_eval=1

model=cnn
num_filters=32
hidden_size=54
num_layers=64

dataset=omniglot
base_dataset_train=train
base_dataset_val=val
base_dataset_test=test
data_path=$DATA_DIR
save_path=$SAVE_DIR


# Log host and date
hostname
date

# Remember to also export the path to the libraries and software needed, example
export PATH=/share/apps/python-3.7.2-shared/bin:/share/apps/julia-1.2.0/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH

# Get into project dir
cd /home/jitfalk/active_meta_learning_py/classification

# Run main script
python main.py --n_runs 5 --n_train_batches 1000 --n_test_batches 100 --tasks_per_metaupdate 16 \
       --tasks_per_metaupdate 16 --evaluate_every 100 --n_way 5 --k_shot 1 --k_query 10 --num_epochs 5 \
       --dataset omniglot --data_path $DATA_DIR --save_path
