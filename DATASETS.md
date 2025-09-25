# Data Preparation

Set Dataset Root (REQUIRED)
```bash
 DATA_ROOT=/path/to/your/datasets
```

## VisDA-C
Download: https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification

Directory structure:
```text
${DATA_ROOT}/
└── VISDA-C/
    ├── train/
    ├── validation/
    ├── train_list.txt
    └── validation_list.txt
```

Training commands:
```bash
# ViT-B/16 backbone
bash scripts/train_VISDA-C_target.sh
# RN101 backbone
bash scripts/train_VISDA-C_target_RN101.sh
```

## DomainNet-126
Download (cleaned version): http://ai.bu.edu/M3SDA/

Directory structure:
```text
${DATA_ROOT}/
└── DomainNet-126/
    ├── real/
    ├── sketch/
    ├── clipart/
    ├── painting/
    ├── real_list.txt
    ├── sketch_list.txt
    ├── clipart_list.txt
    └── painting_list.txt
```

Training command:
```bash
bash scripts/train_domainnet-126_target.sh
```

## VLCS
Download the VLCS dataset and organize as:

Directory structure:
```text
${DATA_ROOT}/
└── VLCS/
    ├── Caltech101/
    ├── LabelMe/
    ├── SUN09/
    ├── VOC2007/
    ├── Caltech101_list.txt
    ├── LabelMe_list.txt
    ├── SUN09_list.txt
    └── VOC2007_list.txt
```

Training commands:
```bash
# ViT-B/16 backbone
bash scripts/train_VLCS_target.sh
# RN101 backbone
bash scripts/train_VLCS_target_RN101.sh
```

## PACS
Download the PACS dataset and organize as:https://dali-dl.github.io/project_iccv2017.html

Directory structure:
```text
${DATA_ROOT}/
└── PACS/
    └── kfold/
        ├── photo/
        ├── art_painting/
        ├── cartoon/
        ├── sketch/
        ├── photo_list.txt
        ├── art_painting_list.txt
        ├── cartoon_list.txt
        └── sketch_list.txt
```

Training command:
```bash
bash scripts/train_PACS_target.sh
```

## OfficeHome
Download: http://hemanthdv.org/OfficeHome-Dataset/

Directory structure:
```text
${DATA_ROOT}/
└── OfficeHome/
    ├── Art/
    ├── Clipart/
    ├── Product/
    ├── Real_World/
    ├── Art_list.txt
    ├── Clipart_list.txt
    ├── Product_list.txt
    └── Real_World_list.txt
```

Training command:
```bash
bash scripts/train_OfficeHome_target.sh
```