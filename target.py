import logging
import os
import time

from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from thop import clever_format

from image_list import ImageList
from utils import (
    adjust_learning_rate,
    concat_all_gather,
    get_augmentation,
    get_Augmixtransforms,
    is_master,
    per_class_accuracy,
    remove_wrap_arounds,
    use_wandb,
    AverageMeter,
    CustomDistributedDataParallel,
    ProgressMeter,
)

from custom_clip_model import get_custom_clip_model
from TrainDef import get_target_optimizer

def print_trainable_params(net, net_name):
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    formatted_params = clever_format([params], "%.4f")
    print(f'Trainable params in {net_name}: {formatted_params}')
    return params  # 

visda_classnames = ['aeroplane', 'bicycle','bus','car','horse','knife','motorcycle','person','plant','skateboard','train','truck']
domainnet_126_classnames = ['aircraft carrier','alarm clock','ant','anvil','asparagus','axe','banana','basket','bathtub','bear','bee','bird','blackberry','blueberry','bottlecap','broccoli','bus','butterfly','cactus','cake','calculator','camel','camera','candle','cannon','canoe','carrot','castle','cat','ceiling fan','cello','cell phone','chair','chandelier','coffee cup','compass','computer','cow','crab','crocodile','cruise ship','dog','dolphin','dragon','drums','duck','dumbbell','elephant','eyeglasses','feather','fence','fish','flamingo','flower','foot','fork','frog','giraffe','goatee','grapes','guitar','hammer','helicopter','helmet','horse','kangaroo','lantern','laptop','leaf','lion','lipstick','lobster','microphone','monkey','mosquito','mouse','mug','mushroom','onion','panda','peanut','pear','peas','pencil','penguin','pig','pillow','pineapple','potato','power outlet','purse','rabbit','raccoon','rhinoceros','rifle','saxophone','screwdriver','sea turtle','see saw','sheep','shoe','skateboard','snake','speedboat','spider','squirrel','strawberry','streetlight','string bean','submarine','swan','table','teapot','teddy-bear','television','The Eiffel Tower','The Great Wall of China','tiger','toe','train','truck','umbrella','vase','watermelon','whale','zebra'] 


@torch.no_grad()
def eval_and_label_dataset(dataloader, model, banks, args):
    wandb_dict = dict()
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()

    # run inference
    logits, gt_labels, indices = [], [], []
    # features = []
    logging.info("Eval and labeling...")
    iterator = tqdm(dataloader) if is_master(args) else dataloader
    for imgs, labels, idxs in iterator:
        imgs = imgs.to("cuda", non_blocking=True)
        logits_cls = model(imgs)

        # features.append(feats)
        logits.append(logits_cls)
        gt_labels.append(labels)
        indices.append(idxs)
            
    # features = torch.cat(features)
    logits = torch.cat(logits)
    gt_labels = torch.cat(gt_labels).to("cuda")
    indices = torch.cat(indices).to("cuda")

    
    if args.distributed:
        # gather results from all ranks
        # features = concat_all_gather(features)
        logits = concat_all_gather(logits)
        gt_labels = concat_all_gather(gt_labels)
        indices = concat_all_gather(indices)

        # remove extra wrap-arounds from DDP
        ranks = len(dataloader.dataset) % dist.get_world_size()
        # features = remove_wrap_arounds(features, ranks)
        logits = remove_wrap_arounds(logits, ranks)
        gt_labels = remove_wrap_arounds(gt_labels, ranks)
        indices = remove_wrap_arounds(indices, ranks)

    assert len(logits) == len(dataloader.dataset)
    pred_labels = logits.argmax(dim=1)
    accuracy = (pred_labels == gt_labels).float().mean() * 100
    logging.info(f"Accuracy of direct prediction: {accuracy:.2f}")
    wandb_dict["Test Acc"] = accuracy

    acc_per_class = per_class_accuracy(
            y_true=gt_labels.cpu().numpy(),
            y_pred=pred_labels.cpu().numpy(),
        )
    acc_per_class_mean = acc_per_class.mean()
    wandb_dict["Test Avg"] = acc_per_class_mean
    wandb_dict["Test Per-class"] = acc_per_class

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(probs)).cuda()
    banks = {
        "probs": probs[rand_idxs][: args.learn.queue_size],
        "ptr": 0,
    }
    if args.refine_label == 'CBPL':
        print('USE Psuedo-labeling Method:',args.refine_label)
        num_classes = probs.shape[1]
        thres = []
        zero_confidence_classes = [] 
        for i in range(num_classes):
            x = probs[pred_labels==i]
            if len(x) < 3:
                threshold = 0
                thres.append(threshold)
                zero_confidence_classes.append(i)
            else:
                a1 =x[:,i]
                a2 = torch.sort(a1,descending=True)[0]
                x_len =len(a2)
                index = round(x_len*args.learn.num_thres) 
                threshold = min(a2[index].item(), args.learn.probs_thres) 
                thres.append(threshold)
        print('thres:',thres)

        # Filter pseudo labels based on calculated thresholds
        mask = torch.zeros_like(pred_labels).bool()
        class_counts = torch.zeros(num_classes, dtype=torch.int64).to(pred_labels.device)

        for c in range(num_classes):
            class_mask = (pred_labels == c) & (probs[:, c] > thres[c])
            mask |= class_mask
            class_counts[c] = class_mask.sum().item()

        # print("\n========== Pseudo-label Statistics by Class ==========")
        # print(f"Total pseudo-label samples: {mask.sum().item()}")
        # for c in range(num_classes):
        #     print(f"Class {c}: {class_counts[c].item()} samples")

        pred_labels = pred_labels[mask] 
        indices = indices[mask]
    else: 
        print('USE Psuedo-labeling Method:', ' Basic Psuedo-labeling')
        mask = probs.max(dim=1).values > args.learn.probs_thres #base pseudo label threshold 0.95
        pred_labels = pred_labels[mask] 
        indices = indices[mask] 
    

    pseudo_item_list = []
    for pred_label, idx in zip(pred_labels, indices):
        img_path, _, img_file = dataloader.dataset.item_list[idx]
        pseudo_item_list.append((img_path, int(pred_label), img_file))
    logging.info(f"Collected {len(pseudo_item_list)} pseudo labels.")

    # 清理
    del logits, probs, pred_labels, mask
    torch.cuda.empty_cache()

    if use_wandb(args):
        wandb.log(wandb_dict)

    return pseudo_item_list, banks, acc_per_class_mean, accuracy


def measure_average_runtime(total_time, epochs):
    return total_time / epochs


def train_target_domain(args):
    total_start_time = time.time()
    logging.info(
        f"Start target training on {args.data.src_domain}-{args.data.tgt_domain}..."
    )

    # if not specified, use the full length of dataset.
    if args.learn.queue_size == -1:
        label_file = os.path.join(
            args.data.image_root, f"{args.data.tgt_domain}_list.txt"
        )
        dummy_dataset = ImageList(args.data.image_root, label_file)
        data_length = len(dummy_dataset)
        args.learn.queue_size = data_length
        del dummy_dataset

    model,model_val = get_custom_clip_model(args)
    model.cuda()
    model_val.cuda()
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = CustomDistributedDataParallel(model,device_ids=[args.gpu])# find_unused_parameters=True
    logging.info(f"1 - Created target model")

    val_transform = get_augmentation("test",args=args)
    label_file = os.path.join(args.data.image_root, f"{args.data.tgt_domain}_list.txt")
    print("-+"*20,"val_transform label_file:",label_file)
    val_dataset = ImageList(
        image_root=args.data.image_root,
        label_file=label_file,
        transform=val_transform,
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, sampler=val_sampler, num_workers=8,prefetch_factor=4
    )
    pseudo_item_list, banks, avg, acc = eval_and_label_dataset(
        val_loader, model_val, banks=None, args=args
    )
    del model_val
    logging.info("2 - Computed initial pseudo labels")

    # Training data
    if args.learn.tpt:
        tpt_transform = get_Augmixtransforms()
        train_dataset = ImageList(
            image_root=args.data.image_root,
            label_file=None,  # uses pseudo labels
            transform=tpt_transform,
            pseudo_item_list=pseudo_item_list,
        )
        train_sampler = DistributedSampler(train_dataset) if args.distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=1, 
            shuffle=(train_sampler is None),
            num_workers=args.data.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=False,
        )
    else:
        train_transform = get_augmentation("moco-v2",args=args)# get_augmentation_versions(args) args=args
        train_dataset = ImageList(
            image_root=args.data.image_root,
            label_file=None,  # uses pseudo labels
            transform=train_transform,
            pseudo_item_list=pseudo_item_list,
        )
        train_sampler = DistributedSampler(train_dataset) if args.distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.data.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.data.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=False,
            persistent_workers=True,  # 保持 worker 进程活跃
            prefetch_factor=4,
        )

    args.learn.full_progress = args.learn.epochs * len(train_loader)
    logging.info("3 - Created train/val loader")

    # define loss function (criterion) and optimizer
    optimizer = get_target_optimizer(model, args)
    logging.info("4 - Created optimizer")

    total_time_train = 0
    total_time_eval = 0
    logging.info("Start training...")

    best_acc=0.0
    best_avg=0.0

    for epoch in range(args.learn.start_epoch, args.learn.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        start_time_train = time.time()
        train_epoch(train_loader, model, banks, optimizer, epoch, args)
        epoch_time_train = time.time() - start_time_train
        total_time_train += time.time() - start_time_train
        hours, remainder = divmod(epoch_time_train, 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info(f"Epoch {epoch} completed in {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")

        start_time_eval = time.time()
        new_pseudo_item_list, _, avg, acc = eval_and_label_dataset(val_loader, model, banks, args)
        total_time_eval += time.time() - start_time_eval

        # if (epoch + 1) % 10 == 0 and epoch < args.learn.epochs - 1:  # 最后一个epoch不需要重建
        #     logging.info(f"Epoch {epoch + 1}: Rebuilding dataset with new pseudo labels...")

        #     train_transform = get_augmentation("moco-v2", args=args)
        #     train_dataset = ImageList(
        #         image_root=args.data.image_root,
        #         label_file=None,
        #         transform=train_transform,
        #         pseudo_item_list=new_pseudo_item_list,
        #     )
        #     train_sampler = DistributedSampler(train_dataset) if args.distributed else None
        #     train_loader = DataLoader(
        #         train_dataset,
        #         batch_size=args.data.batch_size,
        #         shuffle=(train_sampler is None),
        #         num_workers=args.data.workers,
        #         pin_memory=True,
        #         sampler=train_sampler,
        #         drop_last=False,
        #         persistent_workers=True,
        #         prefetch_factor=4,
        #     )
        
        #     args.learn.full_progress = (args.learn.epochs - epoch - 1) * len(train_loader)
        #     logging.info(f"Dataset rebuilt with {len(new_pseudo_item_list)} pseudo labels")
        
        torch.cuda.empty_cache()

        best_acc = max(best_acc, acc)
        best_avg = max(best_avg, avg)

        # wandb记录
        if use_wandb(args):
            wandb.log({
                "accuracy/current_acc": acc,
                "accuracy/best_acc": best_acc,
                "accuracy/current_avg": avg,
                "accuracy/best_avg": best_avg,
            })

    if is_master(args):
        total_training_time = time.time() - total_start_time
        avg_train_time = total_time_train / args.learn.epochs
        avg_eval_time = total_time_eval / args.learn.epochs

        print(args)
        
        logging.info(f"Training completed! Best acc: {best_acc:.2f}%, Best avg: {best_avg:.2f}%")
        logging.info(f"Total training time: {format_time(total_training_time)} ({total_training_time:.2f}s)")
        logging.info(f"Average time per epoch: Train={format_time(avg_train_time)}, Eval={format_time(avg_eval_time)}")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def train_epoch(train_loader, model, banks, optimizer, epoch, args):
    loss_fn = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_psd = AverageMeter("PLabel-Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, top1_psd],
        prefix=f"Epoch: [{epoch}]",
    )

    # make sure to switch to train mode
    model.train()
    if hasattr(model, 'clip_model'):
        model.clip_model.eval()
    elif hasattr(model, 'module') and hasattr(model.module, 'clip_model'):
        model.module.clip_model.eval()

    end = time.time()
    for i, (images, label, idxs) in enumerate(train_loader):
        label = label.cuda()

        if args.learn.tpt:
            images = images.squeeze(0).cuda(non_blocking=True)
            label = label.repeat(images.size(0))
        else:
            images = images.cuda(non_blocking=True)
        
        # Learning rate adjustment
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, args)

        logits_s = model(images)

        if args.learn.tpt:
            logits_s, selected_idx = select_confident_samples(logits_s, 0.1)
            label = label[selected_idx]
            
        loss = loss_fn(logits_s, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        accuracy = calculate_acc(logits_s, label)
        top1_psd.update(accuracy.item(), len(logits_s))
        loss_meter.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if use_wandb(args):
            wandb_dict = {
                "loss_CLS": loss.item(),
                "acc_pseudo_label": accuracy.item(),
                "epoch": epoch,
            }

            wandb.log(wandb_dict, commit=(i== len(train_loader) - 1))

        if i % args.learn.print_freq == 0:
            progress.display(i)
    # print(f"=== Train Epoch {epoch} End ===")
    # print(torch.cuda.memory_summary(), end='')


@torch.no_grad()
def calculate_acc(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean() * 100


def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes
    loss = (-targets * log_probs).sum(dim=1).mean()

    return loss


def cross_entropy_loss(logits, labels, args):
    if args.learn.ce_type == "standard":
        return F.cross_entropy(logits, labels)
    raise NotImplementedError(f"{args.learn.ce_type} CE loss is not implemented.")


def entropy_minimization(logits):
    if len(logits) == 0:
        return torch.tensor([0.0]).cuda()
    probs = F.softmax(logits, dim=1)
    ents = -(probs * probs.log()).sum(dim=1)

    loss = ents.mean()
    return loss

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx