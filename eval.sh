CODE_FOLDER=escolta

export PATH="/home/michon/anaconda2/bin:$PATH"

source /home/michon/anaconda2/bin/activate py35

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64/:/home/michon/cuda/lib64/:/DEV/cuda/lib64/:/home/klein/lib/cuda/lib64/:/home/klein/cuda/lib64/
export CUDA_VISIBLE_DEVICES=$4
data=$3
checkpoint=$1
config_dir=$2
train(){
	echo "############# train @ "`date`" GPUS=$GPUS HOST=$HOST PWD="`pwd`
	python eval.py --config_dir $config_dir --checkpoint $checkpoint --data $data
    echo "############# train: DONE @ "`date` 
}

TIME=`date +"%Y-%m-%d_%H-%M-%S"`

train #&> log.$MODEL.$CORPUS.$TIME &
