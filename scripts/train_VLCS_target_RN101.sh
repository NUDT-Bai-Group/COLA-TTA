

PORT=10001
MEMO="target"
DATA_ROOT="PATH_TO_YOUR_DATASET_ROOT"  # e.g., /path/to/your/dataset


for model in COLA
do
    for SEED in 2021 
    do
        for backbone in 'RN101'
        do
            for target_domains in VOC2007 LabelMe Caltech101 SUN09 
            do 
                python main_COLA.py \
                seed=${SEED} port=${PORT} memo=${MEMO} project="VLCS" \
                data.data_root=${DATA_ROOT} data.workers=12 \
                data.dataset="VLCS" data.source_domains="[CLIP]" data.target_domains="[${target_domains}]" \
                model_src.arch="${model}" model_src.backbone="${backbone}" \
                optim.lr=5e-3 optim.mlp_lr=5e-3 optim.hidden=128 optim.factor=1 optim.weight_decay=1e-4 \
                data.batch_size=128 learn.probs_thres=0.75 learn.num_thres=0.55 \
                learn.epochs=15  2>&1 | tee "VLCS_${model}-hidden128_$(echo ${backbone} | sed 's|/|_|g')_Target_{${target_domains}}_$(date '+%Y-%m-%d_%H_%M_%S').txt" 2>&1
            done
        done
    done
done
