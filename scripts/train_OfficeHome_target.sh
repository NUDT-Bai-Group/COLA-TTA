SRC_DOMAIN=$1
TGT_DOMAIN=$2

PORT=10001
MEMO="target"
DATA_ROOT="PATH_TO_YOUR_DATASET_ROOT"  # e.g., /path/to/your/dataset

for model in COLA 
do
    for SEED in 2021 
    do
        for backbone in 'ViT-B/16' 'RN101'
        do
            for target_domains in Art Clipart Product Real_World
            do 
                python main_COLA.py \
                seed=${SEED} port=${PORT} memo=${MEMO} project="OfficeHome" multiprocessing_distributed='false' \
                data.data_root=${DATA_ROOT} data.workers=12 \
                data.dataset="OfficeHome" data.source_domains="[CLIP]" data.target_domains="[${target_domains}]" \
                model_src.arch="${model}" model_src.backbone="${backbone}" \
                learn.Alpha=1 learn.Beta=1 learn.Gamma=1 \
                optim.lr=1e-1 optim.mlp_lr=5e-2 optim.hidden=128 optim.factor=1 optim.weight_decay=1e-6 \
                data.batch_size=128 learn.probs_thres=0.7 learn.num_thres=0.6 \
                learn.epochs=15  2>&1 | tee "OfficeHome_${model}_$(echo ${backbone} | sed 's|/|_|g')_Target_{${target_domains}}_$(date '+%Y-%m-%d_%H_%M_%S').txt" 2>&1
            done
        done
    done
done
