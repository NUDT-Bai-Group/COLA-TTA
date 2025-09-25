import logging
import os

import torch
import torch.nn.functional as F

import clip

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import json

from thop import clever_format
from collections import OrderedDict


visda_classnames = ['aeroplane', 'bicycle','bus','car','horse','knife','motorcycle','person','plant','skateboard','train','truck']
terra_classnames = ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'background', 'opossum', 'rabbit', 'raccoon', 'squirrel']
VLCS_classnames = ['bird', 'car', 'chair', 'dog', 'person']
PACS_classnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
OfficeHome_classnames = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker_Pen']
domainnet_126_classnames = ['aircraft carrier','alarm clock','ant','anvil','asparagus','axe','banana','basket','bathtub',
                            'bear','bee','bird','blackberry','blueberry','bottlecap','broccoli','bus','butterfly','cactus','cake',
                            'calculator','camel','camera','candle','cannon','canoe','carrot','castle','cat','ceiling fan','cello',
                            'cell phone','chair','chandelier','coffee cup','compass','computer','cow','crab','crocodile','cruise ship',
                            'dog','dolphin','dragon','drums','duck','dumbbell','elephant','eyeglasses','feather','fence','fish',
                            'flamingo','flower','foot','fork','frog','giraffe','goatee','grapes','guitar','hammer','helicopter',
                            'helmet','horse','kangaroo','lantern','laptop','leaf','lion','lipstick','lobster','microphone','monkey',
                            'mosquito','mouse','mug','mushroom','onion','panda','peanut','pear','peas','pencil','penguin','pig','pillow',
                            'pineapple','potato','power outlet','purse','rabbit','raccoon','rhinoceros','rifle','saxophone','screwdriver',
                            'sea turtle','see saw','sheep','shoe','skateboard','snake','speedboat','spider','squirrel','strawberry',
                            'streetlight','string bean','submarine','swan','table','teapot','teddy-bear','television','The Eiffel Tower',
                            'The Great Wall of China','tiger','toe','train','truck','umbrella','vase','watermelon','whale','zebra'] 


def get_prompt(classnames,args):
    if args.data.dataset == "DomainNet-126"or args.data.dataset == "OfficeHome" :
    # if args.data.dataset == "OfficeHome" :
        if "tgt_domain" in args.data and args.data.tgt_domain:
            domain_info = args.data.tgt_domain
            if args.data.tgt_domain == "Real_World":
                domain_info = "Real"
            temp = 'a {} style photo of a {}.'
            prompts = [temp.format(domain_info,c).replace("_", " ") for c in classnames]
        else :
            temp = "An image depicting a {}."
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(prompts)
    elif args.data.dataset == "terra_incognita" :
        temp = "a wildlife snapshot of a {}." #"An image capturing a {} in the wild." #"a photo containing {}." #"An image containing a {}."a picture of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
    else:
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
    prompts = torch.cat([clip.tokenize(p) for p in prompts])
    prompts = prompts.cuda()
    print("Prompts build done!!","-"*20)
    return prompts

class Adapter(nn.Module):
    def  __init__(self,c_in,hidden_size):
        super(Adapter,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in,hidden_size , bias=False),#c_in // reduction
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size,c_in, bias=False),#c_in // reduction
            nn.ReLU(inplace = True)
        )
    def forward(self,x):
        x = self.fc(x)
        return x


class ResMLP(nn.Module):
    def __init__(self,n_in, hidden_size):
        super().__init__()
        self.input = nn.Linear(n_in, hidden_size) 
        self.dropout = nn.Dropout(0.1) 
        self.hiddens = nn.Linear(hidden_size,hidden_size)
        self.output = nn.Linear(hidden_size, n_in)
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self,x):
        residual = x
        out = self.input(x)# 512-->hidden_size
        out = self.dropout(out)
        out = self.relu(out)
        out = self.hiddens(out)# hidden_size-->hidden_size
        out = self.relu(out)
        out = self.output(out)# hidden_size-->512
        out = out + residual * torch.sigmoid(self.scale)
        return out

class BaseClipModel(nn.Module):
    def __init__(self, args = None ,classnames= None ,clip_model=None,prompt = None):
        nn.Module.__init__(self)
        self.args = args
        self.classnames = classnames
        self.clip_model = clip_model
        self.prompt = prompt
        self.dtype = self.clip_model.dtype
        self._output_dim = self.clip_model.visual.output_dim
        self.logit_scale = self.clip_model.logit_scale   

    def forward(self, x, return_feats=False):
        raise NotImplementedError

    def freeze_parameters(self):
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

    def load_from_checkpoint(self, checkpoint_path):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        return len(self.classnames) 

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0

class BaseCLIP(BaseClipModel):
    def __init__(self,classnames ,clip_model,prompt):
        super().__init__(classnames= classnames ,clip_model=clip_model,prompt =prompt)
        for param in self.clip_model.transformer.parameters():
            param.requires_grad_(False)
        
    def forward(self, image, return_feats=False):
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(self.prompt)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp() 
        logits = logit_scale * image_features @ text_features.t()

        if return_feats:
            return image_features, logits
        return logits
    
    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        backbone_params.extend(self.clip_model.visual.parameters())
        extra_params = []

        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

class ClipAdapter(BaseClipModel):
    def __init__(self,classnames, clip_model,prompt,args):
        super().__init__(classnames=classnames, clip_model=clip_model,prompt = prompt)
        self.adapter_s = Adapter(self._output_dim,4).to(self.dtype)
        self.backbone_params = []
        self.extra_params = []
        self.MLP_params = []
        
            
    def forward(self, image, return_feats=False):
        image_features = self.clip_model.encode_image(image)
        x = self.adapter_s(image_features)
        text_features = self.clip_model.encode_text(self.prompt)#[classnum,Embed_dim]

        ratio = 0.2 # CLIP-Adapter default set
        image_features = ratio * x + (1 - ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp() 
        logits = logit_scale * image_features @ text_features.t()

        if return_feats:
            return image_features, logits
        return logits
    
    def get_params(self):

        self.extra_params.extend(self.adapter_s.parameters())
        self.params_ = sum(p.numel() for p in self.extra_params if p.requires_grad)
        self.params__ = clever_format([self.params_], "%.4f")
        print('+-'*10,f'trainable params number: {self.params__}','*^'*10)

        # exclude frozen params
        # self.backbone_params = [param for param in self.backbone_params if param.requires_grad]
        # self.extra_params = [param for param in self.extra_params if param.requires_grad]

        return self.backbone_params, self.extra_params, self.MLP_params

class COLA(BaseClipModel):
    def __init__(self, classnames ,clip_model,prompt,args):
        super().__init__(classnames= classnames ,clip_model=clip_model,prompt =prompt)
        self.MLP_params = []
        self.adapter_params = []
        self.prompt =prompt

        self.clip_ratio = args.learn.Alpha
        self.adapter_ratio = args.learn.Beta
        self.domain_ratio = args.learn.Gamma

        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.prompt)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.register_buffer('cached_text_features', text_features)
        
        # self.adapter = Adapter(self._output_dim,4).to(self.dtype)
        self.adapter = Adapter(self._output_dim,args.optim.hidden).to(self.dtype)
        self.network = ResMLP(self._output_dim,args.optim.hidden*args.optim.factor).to(dtype=self.clip_model.dtype)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        # self.adapter_s.apply(init_weights)
        self.network.apply(init_weights)
        self.num_classes_ = len(classnames)
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        
        del clip_model

    def forward(self,image,return_feats=False,CLIP_OUT=False):
        with torch.no_grad():
            raw_image_features = self.clip_model.encode_image(image)
        text_features = self.cached_text_features
        
        batch_size = raw_image_features.size(0)
        mean_domain_features = raw_image_features.mean(dim=0, keepdim=True)
        domain_features = self.network(mean_domain_features)
        domain_features = domain_features.expand(batch_size, -1)  # 显式扩展

        x = self.adapter(raw_image_features)
        combined_image_features =(self.clip_ratio * raw_image_features) + (self.adapter_ratio * x)  + (self.domain_ratio * domain_features)
        combined_image_features = combined_image_features.type(self.dtype)
        
        normalized_image_features = combined_image_features / combined_image_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * normalized_image_features @ text_features.t()
        
        if CLIP_OUT:
            normalized_CLIP_image_features = raw_image_features / raw_image_features.norm(dim=-1, keepdim=True)
            logits_CLIP = self.clip_model.logit_scale.exp() * normalized_CLIP_image_features @ text_features.t()
            return logits_CLIP, logits_per_image
        if return_feats:
            return normalized_image_features, logits_per_image
        return logits_per_image


    def load_from_checkpoint_CAM(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            if "clip_model" not in name:
                if "adapter_s" in name or "network" in name:
                    state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):

        self.MLP_params = list(self.network.parameters())
        self.adapter_params = list(self.adapter.parameters()) 

        # MLP params
        params_M = sum(p.numel() for p in self.MLP_params if p.requires_grad)
        params_MLP = clever_format([params_M], "%.4f")
        print('|+-+|'*10, f'trainable MLP_params params number: {params_MLP}', '|*^*|'*10)

        # adapter params
        params_ = sum(p.numel() for p in self.adapter_params if p.requires_grad)
        params__ = clever_format([params_], "%.4f")
        print('|+-+|'*10, f'trainable adapter_params number: {params__}', '|*^*|'*10)

        # exclude frozen params
        self.MLP_params = [param for param in self.MLP_params if param.requires_grad]
        self.adapter_params = [param for param in self.adapter_params if param.requires_grad]

        return self.adapter_params, self.MLP_params
    

class CoOp_TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding #torch.size([77,512])
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND float32
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.clip_model = clip_model
        n_cls = len(classnames)
        n_ctx = 16  # cfg.TRAINER.COOP.N_CTX #number of context vectors
        ctx_init =  "a photo of a" # cfg.TRAINER.COOP.CTX_INIT ## initialization words
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution#224
        # cfg_imsize = 224  #  cfg.INPUT.SIZE[0]#config/*.yaml
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))#len(['a', 'photo', 'of', 'a'])  4
        prompt = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        self.prompt_prefix = ctx_init

        # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        # nn.init.normal_(ctx_vectors, std=0.02)
        # self.prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # print('ctx.shape',self.ctx.shape)#[16,512]

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()#torch.Size([12, 77])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            #torch.Size([12, 77, 512])


        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end" # cfg.TRAINER.COOP.CLASS_TOKEN_POSITION # 'middle' or 'end' or 'front'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ], 
            dim=1,
        ) 

        return prompts
    
    def update_classnames(self, new_classnames):
        if new_classnames == self.classnames:
            return
        self.classnames = [name.replace("_", " ") for name in new_classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in self.classnames]
        self.prompts = [self.prompt_prefix + " " + name + "." for name in self.classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in self.prompts]).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
    
class CoPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.clip_model = clip_model
        n_ctx = 16 #cfg.TRAINER.COCOOP.N_CTX
        ctx_init = "a photo of a" #cfg.TRAINER.COCOOP.CTX_INIT
        # PREC ='fp16' #cfg.TRAINER.COCOOP.PREC
        self.dtype = self.clip_model.dtype
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        vis_dim = self.clip_model.visual.output_dim
        clip_imsize = self.clip_model.visual.input_resolution
        # cfg_imsize = 224  #  cfg.INPUT.SIZE[0]#config/*.yaml
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        self.prompt_prefix = ctx_init

        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        # if PREC == "fp16":
        self.meta_net.half()
        

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def update_classnames(self, classnames):
        self.n_cls = len(classnames)
        self.classnames = classnames

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS


    def construct_prompts(self, ctx, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts

class CoOpCLIP(BaseClipModel):
    def __init__(self, classnames, clip_model):
        super().__init__(classnames= classnames ,clip_model=clip_model)
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = CoOp_TextEncoder(clip_model)
        self.backbone_params = []
        self.extra_params = []
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def forward(self, image,return_feats=False):
        image_features = self.clip_model.encode_image(image)
        prompts = self.prompt_learner() 
        tokenized_prompts = self.tokenized_prompts #int32#torch.Size([12, 77])
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if return_feats:
            return image_features, logits
        return logits
    
    def get_params(self):

        self.extra_params.extend(self.prompt_learner.parameters())

        # exclude frozen params
        self.backbone_params = [param for param in self.backbone_params if param.requires_grad]

        self.extra_params = [param for param in self.extra_params if param.requires_grad]

        return self.backbone_params, self.extra_params

class CoCoOpCLIP(BaseClipModel):
    def __init__(self, classnames, clip_model):
        super().__init__(classnames= classnames ,clip_model=clip_model)
        self.prompt_learner = CoPromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = CoOp_TextEncoder(clip_model)
        self.backbone_params = []
        self.extra_params = []
        self.MLP_params = []
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

    def forward(self, image,return_feats=False):
        logit_scale = self.logit_scale.exp()
        tokenized_prompts = self.tokenized_prompts #int32#torch.Size([12, 77])
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        #prompts shape: torch.Size([64, 12, 77, 512])
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            # with torch.no_grad():# 
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        if return_feats:
            return image_features, logits
        return logits
    
    def get_params(self):

        self.extra_params.extend(self.prompt_learner.parameters())

        # exclude frozen params
        self.backbone_params = [param for param in self.backbone_params if param.requires_grad]

        self.extra_params = [param for param in self.extra_params if param.requires_grad]

        return self.backbone_params, self.extra_params




def get_custom_clip_model(args):
    print('args.data.dataset:',args.data.dataset)

    print('###'*10)
    if args.data.dataset == 'VISDA-C':
        classnames = visda_classnames
        print('get "',args.data.dataset,'" classnames')

    elif args.data.dataset == 'DomainNet-126':
        classnames = domainnet_126_classnames
        print('get "',args.data.dataset,'" classnames')

    elif args.data.dataset == 'terra_incognita':
        classnames = terra_classnames

    elif args.data.dataset == 'VLCS':
        classnames = VLCS_classnames
        print('get "',args.data.dataset,'" classnames')

    elif args.data.dataset == 'PACS':
        classnames = PACS_classnames

        print('get "',args.data.dataset,'" classnames')

    elif args.data.dataset == 'OfficeHome':
        classnames = OfficeHome_classnames
        print('get "',args.data.dataset,'" classnames')

    backbone = args.model_src['backbone']

    print(f"Loading CLIP (backbone: {backbone})")
    clip_model, preprocess = clip.load(backbone)
    clip_model.cuda()

    if args.model_src['arch'] == 'BaseCLIP' or args.model_src['arch'] == "CLIP_TPT":
        with torch.no_grad():
            prompts = get_prompt(classnames,args)
        model = BaseCLIP(classnames, clip_model, prompts)
        model_val = BaseCLIP(classnames, clip_model, prompts)
        print('Initialize the BaseCLIP model ')
    
    elif args.model_src['arch'] == 'ClipAdapter': 
        with torch.no_grad():
            prompts = get_prompt(classnames,args)
        model = ClipAdapter(classnames, clip_model,prompts,args).cuda()#
        model_val = BaseCLIP(classnames, clip_model,prompts)
        model.freeze_parameters()
        print("freeze CLIP parameters ")
        print('(*=*)'*5,'CLIP_output_dim:',model.output_dim,'(*=*)'*5)
        print('(*=*)'*5,'ResMLP hidden size:',args.optim.hidden,'(*=*)'*5)
        print('(*=*)'*5,'Adapter LR:',args.optim.lr,'(*=*)'*5)
        print('Initialize the ClipAdapter model ')

    elif args.model_src['arch'] == 'COLA':
        with torch.no_grad():
            prompts = get_prompt(classnames,args)
        model = COLA(classnames, clip_model,prompts,args).cuda()#
        model_val = BaseCLIP(classnames, clip_model,prompts)
        model.freeze_parameters()
        print("freeze CLIP parameters ")
        print('(*=*)'*5,'CLIP_output_dim:',model.output_dim,'(*=*)'*5)
        print('(*=*)'*5,'ResMLP hidden size:',args.optim.hidden,'(*=*)'*5)
        print('(*=*)'*5,'Adapter LR:',args.optim.lr,'(*=*)'*5)
        print('(*=*)'*5,'ResMLP LR:',args.optim.mlp_lr,'(*=*)'*5)
        print('Initialize the COLA model ')
    
    elif args.model_src['arch'] == 'CoOpCLIP': 
        with torch.no_grad():
            prompts = get_prompt(classnames,args)
        model = CoOpCLIP(classnames, clip_model) 
        model_val = BaseCLIP(classnames, clip_model,prompts)
        print('(*=*)'*5,'CoOp LR:',args.optim.lr,'(*=*)'*5)
        print('Initialize the CoOpCLIP model ')

    elif args.model_src['arch'] == 'CoCoOpCLIP': 
        with torch.no_grad():
            prompts = get_prompt(classnames,args)
        model = CoCoOpCLIP(classnames, clip_model) 
        model_val = BaseCLIP(classnames, clip_model,prompts)
        print('(*=*)'*5,'CoCoOp LR:',args.optim.lr,'(*=*)'*5)
        print('Initialize the CoCoOpCLIP model ')

    return model,model_val