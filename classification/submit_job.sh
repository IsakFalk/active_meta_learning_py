#$ -l tmem=8GB
#$ -l h_rt=4:0:0

#$ -S /bin/bash
#$ -j y
#$ -N aml_omniglot

#$ -l gpu=true

# Scratch for data IO
#$ -l tscratch=4G
mkdir -p /scratch0/jitfalk/$JOB_ID

# Run main script
python main.py --n_runs 5 --n_train_batches 1000 --n_test_batches 100 --tasks_per_metaupdate 16 \
       --tasks_per_metaupdate 16 --evaluate_every 100 --n_way 5 --k_shot 1 --k_query 10 --num_epochs 5 \
       --dataset omniglot --data_path /scratch0/jitfalk/$JOB_ID --save_path /home/jitfalk

# Exit statement
function finish {
    rm -rf /scratch0/jitfalk/$JOB_ID
}

trap finish EXIT ERR
