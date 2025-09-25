import torch

def get_target_optimizer(model, args):
    if args.distributed:
        model = model.module
    if hasattr(model, "src_model"):
        params = model.src_model.get_params()
    else:
        params = model.get_params()
    if len(params) == 3:
        backbone_params, adapter_params,mlp_params = params
    else:
        adapter_params, mlp_params = params
        backbone_params = []

    # for param in extra_params:
    #     print('extra_params',param.is_leaf)
    #     print(param)
    # for i, param in enumerate(extra_params):
    #     if not param.is_leaf:
    #         print(f'extra_params[{i}] is not a leaf tensor')


    if args.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": adapter_params,
                    "lr": args.optim.lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
                {
                    "params": mlp_params,
                    "lr": args.optim.mlp_lr,
                    "momentum": args.optim.momentum,
                    "weight_decay": args.optim.weight_decay,
                    "nesterov": args.optim.nesterov,
                },
            ]
        )
    elif args.optim.name == "adam":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": backbone_params,
                    "lr": args.optim.lr,
                    "betas":(0.9, 0.999),
                    "weight_decay": args.optim.weight_decay,
                },
                {
                    "params": adapter_params,
                    "lr": args.optim.lr,
                    "betas":(0.9, 0.999),
                    "weight_decay": args.optim.weight_decay,
                },
                {
                    "params": mlp_params,
                    "lr": args.optim.lr,
                    "betas":(0.9, 0.999),
                    "weight_decay": args.optim.weight_decay,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{args.optim.name} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer
