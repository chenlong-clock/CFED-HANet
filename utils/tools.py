import json, os
import torch
from configs import parse_arguments
args = parse_arguments()
device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")  # type: ignore

def compute_CLLoss(Adj_mask, reprs, matsize): # compute InfoNCELoss
    logits_cl = torch.div(torch.matmul(reprs, reprs.T), args.cl_temp)
    if args.sub_max:
        logits_max_cl, _ = torch.max(logits_cl, dim=-1, keepdim=True)
        logits_cl = logits_cl - logits_max_cl
    exp_logits_cl = torch.exp(logits_cl)
    denom_cl = torch.sum(exp_logits_cl * (1 - torch.eye(matsize).to(device)), dim = -1) 
    log_prob_cl = -torch.mean((logits_cl - torch.log(denom_cl)) * Adj_mask, dim=-1)
    return torch.mean(log_prob_cl[log_prob_cl > 0])

def collect_from_json(dataset, root, split):
    default = ['train', 'dev', 'test']
    if split == "train":
        pth = os.path.join(root, dataset, "perm"+str(args.perm_id), f"{dataset}_{args.task_num}task_{args.class_num // args.task_num}way_{args.shot_num}shot.{split}.jsonl")
    elif split in ['dev', 'test']:
        pth = os.path.join(root, dataset, f"{dataset}.{split}.jsonl")
    elif split == "stream":
        pth = os.path.join(root, dataset, f"stream_label_{args.task_num}task_{args.class_num // args.task_num}way.json")
    else:
        raise ValueError(f"Split \"{split}\" value wrong!")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Path {pth} do not exist!")
    else:
        with open(pth) as f:
            if pth.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
                if split == "train":
                    data = [list(i.values()) for i in data]
            else:
                data = json.load(f)
    return data