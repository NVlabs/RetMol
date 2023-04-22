

GPUS_PER_NODE=4 # 4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6699
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
# config_json="$script_dir/megatron_molbart/ds_config.json"
config_json="megatron_molbart/ds_config.json"

#ZeRO Configs
stage=1
reduce_scatter=true
contigious_gradients=false
rbs=50000000
agbs=5000000000

chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false

# Megatron Model Parallelism
mp_size=1
# DeepSpeed Pipeline parallelism
pp_size=0


#######
## JACKMOD: add two options: 1 for data, 1 for tensorboard
megatron_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --num-layers 4 \
        --hidden-size 256 \
        --num-attention-heads 8 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --batch-size 320 \
        --gas 16 \
        --train-iters 50000 \
        --lr-decay-iters 320000 \
        --data-impl mmap \
        --distributed-backend nccl \
        --lr 0.0001 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 0 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 1 \
        --save-interval 10000 \
        --eval-interval 50000 \
        --eval-iters 10 \
        --fp16
        --dataset_path ../data/zinc.tab 
"
#         --tensorboard-dir tensorboard_v4_attr-logp-sa
#         --save megatron_molbart_v4_attr-logp-sa_checkpoint
 

deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${megatron_options} ${deepspeed_options} ${chkp_opt}"

custom_train_options=" \
                --stage 1 \
                --train_from pretrain \
                --model_ckpt_itr 50000 \
                --attr logp-sa \
                --attr_offset 0 \
                --data_source jtnn \
                --enumeration_input false \
                --retriever_rule random \
                --pred_target reconstruction \
                --n_retrievals 10 \
                --n_neighbors 100
                "

# run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --master_port=${MASTER_PORT} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} \
#         megatron_molbart/train_retrieval.py $@ ${full_options} ${custom_train_options}"
run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --master_port=${MASTER_PORT} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} \
        train_retrieval.py ${full_options} ${custom_train_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
