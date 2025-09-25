import os

# 给定的类别列表
category_list = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

# 用于将类别映射到数字标签的字典
category_mapping = {category_list[i]: i for i in range(len(category_list))}

# 获取当前脚本文件所在的目录
script_dir = '/media/ubuntu/D/code/base-clip/code/AdaContrast-master/datasets/PACS'
dataset_root = script_dir
output_dir = script_dir

# 指定的域列表
specified_domains = ['art_painting', 'cartoon', 'photo', 'sketch']

for domain in specified_domains:
    domain_path = os.path.join(dataset_root, 'kfold', domain)
    output_filename = f"{domain}_list.txt"
    output_file_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(domain_path):
        with open(output_file_path, 'w') as output_file:
            for category in category_list:
                category_path = os.path.join(domain_path, category)
                category_label = category_mapping.get(category)
                
                if category_label is not None and os.path.exists(category_path):
                    for image_file in os.listdir(category_path):
                        if image_file.endswith('.jpg') or image_file.endswith('.png'):
                            image_path = os.path.join(category_path, image_file)
                            # 获取相对路径并写入文件
                            relative_path = os.path.relpath(image_path, start=dataset_root)
                            output_file.write(f"{relative_path} {category_label}\n")
    else:
        print(f"指定的域 '{domain}' 不存在于数据集中。")
