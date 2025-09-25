SRC_MODEL_DIR=$1

PORT=10000
MEMO="target"
DATA_ROOT="PATH_TO_YOUR_DATASET_ROOT"  # e.g., /path/to/your/dataset

for SEED in 2021
do
    for model in COLA 
    do
        for backbone in 'RN101' 'ViT-B/16'
        do
            for target_domains in real clipart sketch painting 
            do
                for probs_thres in 0.85
                do
                    for num_thres in  0.35 
                    do
                        python main_COLA.py \
                        seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet-126"  \
                        optim.lr="5e-2" optim.mlp_lr="1e-2" optim.hidden=128 optim.weight_decay="1e-4" \
                        learn.probs_thres=${probs_thres} learn.num_thres=${num_thres} \
                        data.data_root=${DATA_ROOT} data.workers=8 \
                        learn.Alpha=0.5 learn.Beta=0.5 learn.Gamma=0.1 \
                        data.dataset="DomainNet-126" data.source_domains="[CLIP]" data.target_domains="[${target_domains}]" \
                        model_src.arch="${model}" model_src.backbone="${backbone}" 2>&1 | tee "DN126_${model}_Hidden_128_Ratio(0.5-0.5-1)_$(echo ${backbone} | sed 's|/|_|g')_Target_{${target_domains}}_thres(${probs_thres}-${num_thres})_$(date '+%Y-%m-%d_%H_%M_%S').txt" 2>&1
                    done
                done
            done
        done
    done
done
