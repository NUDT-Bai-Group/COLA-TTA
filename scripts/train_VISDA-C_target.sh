SRC_MODEL_DIR=$1

PORT=10001
MEMO="target"
DATA_ROOT="PATH_TO_YOUR_DATASET_ROOT"  # e.g., /path/to/your/dataset

for SEED in 2021
do
    for model in COLA 
    do
        for backbone in "ViT-B/16"
        do
            for LR in 5e-4
            do
                for mlp_lr in 5e-5 
                do
                    for probs_thres in 0.95
                    do
                        for num_thres in 0.85
                        do
                            for hidden_size in 128 
                            do
                                python main_COLA.py \
                                seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
                                optim.lr=${LR} optim.mlp_lr=${mlp_lr} optim.hidden=${hidden_size} optim.weight_decay="1e-6" \
                                learn.probs_thres=${probs_thres} learn.num_thres=${num_thres} \
                                data.data_root=${DATA_ROOT} data.workers=4 \
                                learn.Alpha=0.1 learn.Beta=0.5 learn.Gamma=1 \
                                data.dataset="VISDA-C" data.source_domains="[CLIP]" data.target_domains="[validation]" \
                                model_src.arch="${model}" model_src.backbone="${backbone}" 2>&1 | tee "VISDA-C_${model}_$(echo ${backbone} | sed 's|/|_|g')-Ratio(0.5-0.1-0.1)-L2-hidden-${hidden_size}_LR-${LR}_MLP-${mlp_lr}_Thres(${probs_thres}-${num_thres})_Target_{${target_domains}}_$(date '+%Y-%m-%d_%H_%M_%S').txt" 2>&1
                            done
                        done
                    done
                done
            done
        done
    done
done

#   learn.Alpha=0.5 learn.Beta=0.1 learn.Gamma=0.1 FOR COLA