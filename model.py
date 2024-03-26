import torch
from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
from configs import parse_arguments
from torch.nn.utils.rnn import unpad_sequence
from random import shuffle

args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")  # type: ignore


class BertED(nn.Module):
    def __init__(self, class_num=args.class_num + 1, input_map=False):
        super().__init__()
        self.backbone = BertModel.from_pretrained(args.backbone)
        if not args.no_freeze_bert:
            print("Freeze bert parameters")
            for _, param in list(self.backbone.named_parameters()):
                param.requires_grad = False
        else:
            print("Update bert parameters")
        self.is_input_mapping = input_map
        self.input_dim = self.backbone.config.hidden_size
        self.fc = nn.Linear(self.input_dim, class_num)
        if self.is_input_mapping:
            self.map_hidden_dim = 512 # 512 is implemented by the paper
            self.map_input_dim =  self.input_dim * 2
            self.input_map = nn.Sequential(
                nn.Linear(self.map_input_dim, self.map_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.map_hidden_dim, self.map_hidden_dim),
                nn.ReLU(),
            )
            self.fc = nn.Linear(self.map_hidden_dim, class_num)

    def forward(self, x, masks, span=None, aug=None):
        # x = self.backbone(x) #TODO: test use
        return_dict = {}
        backbone_output = self.backbone(x, attention_mask = masks)
        x, pooled_feat = backbone_output[0], backbone_output[1]
        context_feature = x.view(-1, x.shape[-1])
        return_dict['reps'] = x[:, 0, :].clone()
        if span != None:
            outputs, trig_feature = [], []
            for i in range(len(span)):
                if self.is_input_mapping:
                    x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
                    x_cdt = x_cdt.permute(1, 0, 2)
                    x_cdt = x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
                    opt = self.input_map(x_cdt)
                else:
                    opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
                    # x = x_cdt.permute(1, 0, 2) 
                trig_feature.append(opt)
            trig_feature = torch.cat(trig_feature)
        outputs = self.fc(trig_feature)
        return_dict['outputs'] = outputs
        return_dict['context_feat'] = context_feature
        return_dict['trig_feat'] = trig_feature
        # if args.single_label:
        #     return_outputs = self.fc(enc_out_feature).view(-1, args.class_num + 1)
        # else:
        #     return_outputs = self.fc(feature)
        if aug is not None:
            feature_aug = trig_feature + torch.randn_like(trig_feature) * aug
            outputs_aug = self.fc(feature_aug)
            return_dict['feature_aug'] = feature_aug
            return_dict['outputs_aug'] = outputs_aug
        return return_dict

    def forward_backbone(self, x, masks):
        x = self.backbone(x, attention_mask = masks)
        x = x.last_hidden_state
        return x

    def forward_input_map(self, x):
        return self.input_map(x)

    # def forward_cl(self, x, masks, span, span_len):
    #     tk_len = torch.count_nonzero(masks, dim=-1) - 2
    #     perm = [torch.randperm(item).to(device) + 1 for item in tk_len]
    #     span = unpad_sequence(span, span_len, batch_first=True)            
    #     if args.scl == "shuffle":
    #         for i in range(len(tk_len)):
    #             span[i] = torch.where(span[i].unsqueeze(2) == perm[i].unsqueeze(0).unsqueeze(0))[2].view(-1, 2) + 1
    #             x[i, 1: 1+tk_len[i]] = x[i, perm[i]]
    #     elif args.scl =="RTR":
    #         rand_ratio = 0.25
    #         rand_num = (rand_ratio * tk_len).int()
    #         special_ids = [103, 102, 101, 100, 0]
    #         all_ids = torch.arange(self.backbone.config.vocab_size).to(device)
    #         special_token_mask = torch.ones(self.backbone.config.vocab_size).to(device)
    #         special_token_mask[special_ids] = 0
    #         all_tokens = all_ids.index_select(0, special_token_mask.nonzero().squeeze())
    #         for i in range(len(rand_num)):
    #             token_idx = torch.arange(tk_len[i]).to(device) + 1
    #             trig_mask = torch.ones(token_idx.shape).to(device)
    #             span_pos = span[i].view(-1).unique() - 1
    #             trig_mask[span_pos] = 0
    #             token_idx_ntrig = token_idx.index_select(0, trig_mask.nonzero().squeeze())
    #             replace_perm = torch.randperm(token_idx_ntrig.shape.numel())
    #             replace_idx = token_idx_ntrig[replace_perm][:rand_num[i]]

    #             new_tkn_idx = torch.randperm(len(all_tokens))[:rand_num[i]]

    #             x[i, replace_idx] = all_tokens[new_tkn_idx].to(device)
    #     backbone_output = self.backbone(x, attention_mask = masks)
    #     if args.cls_reps:
    #         x = backbone_output[0]
    #         reps = x[:, 0, :].clone()
    #     else:
    #         x = backbone_output[0]
    #         reps = backbone_output[1]
    #     enc_out_feature = x
    #     # x = x.[:, 0, :].squeeze(1)
    #     # x = self.ContrastMLP(pooled_x)
    #     outputs, feature = [], []
    #     for i in range(len(span)):
    #         if self.is_input_mapping:
    #             x_cdt = torch.stack([torch.index_select(x[i], 0, span[i][:, j]) for j in range(span[i].size(-1))])
    #             x_cdt = x_cdt.permute(1, 0, 2)
    #             x_cdt = x_cdt.contiguous().view(x_cdt.size(0), x_cdt.size(-1) * 2)
    #             opt = self.input_map(x_cdt)
    #         else:
    #             opt = torch.index_select(x[i], 0, span[i][:, 0]) + torch.index_select(x[i], 0, span[i][:, 1])
    #             # x = x_cdt.permute(1, 0, 2) 
    #         feature.append(opt)
    #     feature = torch.cat(feature)
    #     outputs = self.fc(feature)
    #     return reps, outputs, enc_out_feature

    # def forward_ace(self, x, masks):
    #     return_dict = {}
    #     backbone_output = self.backbone(x, attention_mask = masks)
    #     x = backbone_output[0]

    #     if args.cls_reps:
    #         reps = x[:, 0, :].clone()
    #     else:
    #         reps = backbone_output[1]
    #     feature = x
    #     outputs = self.fc(feature)
    #     return_dict['outputs'] = outputs
    #     return_dict['feature'] = feature
    #     return_dict['reps'] = reps
    #     return return_dict