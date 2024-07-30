import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from torch.optim import AdamW
from utils import *
from configs import parse_arguments
from model import BertED
from tqdm import tqdm
from exemplars import Exemplars
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter   
import os, time
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



# PERM_5 = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

# PERM_10 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]



def train(local_rank, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    if args.log:
        if not os.path.exists(os.path.join(args.tb_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id))):
            os.makedirs(os.path.join(args.tb_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id)))
        if not os.path.exists(os.path.join(args.log_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id))):
            os.makedirs(os.path.join(args.log_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id)))
        writer = SummaryWriter(os.path.join(args.tb_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id), cur_time))
        fh = logging.FileHandler(os.path.join(args.log_dir, args.dataset, args.joint_da_loss, str(args.class_num) + "class", str(args.shot_num) + "shot", args.cl_aug, args.log_name, "perm" + str(args.perm_id), cur_time + '.log'), mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    for arg in vars(args):
        logger.info('{}={}'.format(arg.upper(), getattr(args, arg)))
    logger.info('')
    # set device, whether to use cuda or cpu
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")  # type: ignore
    # get streams from json file and permute them in pre-defined order
    # PERM = PERM_5 if args.task_num == 5 else PERM_10
    streams = collect_from_json(args.dataset, args.stream_root, 'stream')
    # streams = [streams[l] for l in PERM[int(args.perm_id)]] # permute the stream
    label2idx = {0:0}
    for st in streams:
        for lb in st:
            if lb not in label2idx:
                label2idx[lb] = len(label2idx)
    streams_indexed = [[label2idx[l] for l in st] for st in streams]
    model = BertED(args.class_num+1, args.input_map) # define model
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay, eps=args.adamw_eps, betas=(0.9, 0.999)) #TODO: Hyper parameters
    # if args.amp:
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") 
    if args.parallel == 'DDP':
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=local_rank, world_size=args.world_size)
        # device_id = [0, 1, 2, 3, 4, 5, 6, 7]
        model = DDP(model, device_ids= [local_rank], find_unused_parameters=True)
    elif args.parallel == 'DP':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7' 
        model = nn.DataParallel(model, device_ids=[int(it) for it in args.device_ids.split(" ")])


    # optimizer = SGD(model.parameters(), lr=args.lr) # TODO: Use AdamW, GPU out of memory

    criterion_ce = nn.CrossEntropyLoss()
    criterion_fd = nn.CosineEmbeddingLoss()
    all_labels = []
    all_labels = list(set([t for stream in streams_indexed for t in stream if t not in all_labels]))
    task_idx = [i for i in range(len(streams_indexed))]
    labels = all_labels.copy()

    # training process
    learned_types = [0]
    prev_learned_types = [0]
    dev_scores_ls = []
    exemplars = Exemplars() # TODO: 
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        task_idx = task_idx[state_dict['stage']:]
        # TODO: test use
        labels = state_dict['labels']
        learned_types = state_dict['learned_types']
        prev_learned_types = state_dict['prev_learned_types']
    if args.early_stop:
        e_pth = "./outputs/early_stop/" + args.log_name + ".pth"
    for stage in task_idx:
        # if stage > 0:
        #     break
        logger.info(f"Stage {stage}")
        logger.info(f'Loading train instances for stage {stage}')
        # stage = 1 # TODO: test use
        # exemplars = Exemplars() # TODO: test use
        if args.single_label:
            stream_dataset = collect_sldataset(args.dataset, args.data_root, 'train', label2idx, stage, streams[stage])
        else:
            stream_dataset = collect_dataset(args.dataset, args.data_root, 'train', label2idx, stage, [i for item in streams[stage:] for i in item])
        if args.parallel == 'DDP':
            stream_sampler = DistributedSampler(stream_dataset, shuffle=True)
            org_loader = DataLoader(
                dataset=stream_dataset,
                sampler=stream_sampler,
                batch_size=args.batch_size,
                # batch_size=args.shot_num + int(args.class_num / args.shot_num),
                collate_fn= lambda x:x)
        else:
            org_loader = DataLoader(
                dataset=stream_dataset,
                shuffle=True,
                batch_size=args.batch_size,
                # batch_size=args.shot_num + int(args.class_num / args.shot_num),
                collate_fn= lambda x:x)
        stage_loader = org_loader
        if stage > 0:
            if args.early_stop and no_better == args.patience:
                logger.info("Early stopping finished, loading stage: " + str(stage))
                model.load_state_dict(torch.load(e_pth))
            prev_model = deepcopy(model) # TODO:test use
            for item in streams_indexed[stage - 1]:
                if not item in prev_learned_types:
                    prev_learned_types.append(item)
            # TODO: test use
            # prev_model = deepcopy(model) # TODO: How does optimizer distinguish deep copy parameters
            # exclude_none_labels = [t for t in streams_indexed[stage - 1] if t != 0]
            logger.info(f'Loading train instances without negative instances for stage {stage}')
            exemplar_dataset = collect_exemplar_dataset(args.dataset, args.data_root, 'train', label2idx, stage-1, streams[stage-1])
            exemplar_loader = DataLoader(
                dataset=exemplar_dataset,
                batch_size=64,
                shuffle=True,
                collate_fn=lambda x:x)
            # exclude_none_loader = train_ecn_loaders[stage - 1]
            # TODO: test use
            # exemplars.set_exemplars(prev_model.to('cpu'), exclude_none_loader, len(learned_types), device)
            exemplars.set_exemplars(prev_model, exemplar_loader, len(learned_types), device)
            # if not args.replay:
            if not args.no_replay:
                stage_loader = exemplars.build_stage_loader(stream_dataset)
            # else:
            #     e_loader = list(exemplars.build_stage_loader(MAVEN_Dataset([], [], [], [])))
            if args.rep_aug != "none":

                e_loader = exemplars.build_stage_loader(MAVEN_Dataset([], [], [], []))
            # prev_model.to(args.device)   # TODO: test use

        for item in streams_indexed[stage]:
            if not item in learned_types:
                learned_types.append(item)
        logger.info(f'Learned types: {learned_types}')
        logger.info(f'Previous learned types: {prev_learned_types}')
        dev_score = None
        no_better = 0
        for ep in range(args.epochs):
            if stage == 0 and args.skip_first:
                continue
            logger.info('-' * 100)
            logger.info(f"Stage {stage}: Epoch {ep}")
            logger.info("Training process")
            model.train()
            logger.info("Training batch:")
            iter_cnt = 0
            for bt, batch in enumerate(tqdm(stage_loader)):
                iter_cnt += 1
                optimizer.zero_grad()
                # if args.single_label:
                #     train_x, train_y, train_masks, train_span = zip(*batch)
                #     y = [[0] * len(train_x[0]) for _ in train_x]
                #     for i in range(len(train_span)):
                #         for j in range(len(train_span[i])):
                #             y[i][train_span[i][j][0]] = train_y[i][j]
                #     train_x = torch.LongTensor(train_x).to(device)
                #     train_masks = torch.LongTensor(train_masks).to(device)
                #     outputs, feature = model(train_x, train_masks)
                #     logits = outputs[:, learned_types]
                #     y = torch.LongTensor(y).to(device)
                #     loss_ce = criterion_ce(logits, y.view(-1))
                #     padded_train_span, span_len = None, None
                # else:
                train_x, train_y, train_masks, train_span = zip(*batch)
                train_x = torch.LongTensor(train_x).to(device)
                train_masks = torch.LongTensor(train_masks).to(device)
                train_y = [torch.LongTensor(item).to(device) for item in train_y]
                train_span = [torch.LongTensor(item).to(device) for item in train_span]
                # if args.dataset == "ACE":
                #     return_dict = model(train_x, train_masks)
                # else:
                return_dict = model(train_x, train_masks, train_span)
                outputs, context_feat, trig_feat = return_dict['outputs'], return_dict['context_feat'], return_dict['trig_feat']
                # invalid_mask_op = torch.BoolTensor([item not in learned_types for item in range(args.class_num)]).to(device)
                # not from below's codes
                for i in range(len(train_y)):
                    invalid_mask_label = torch.BoolTensor([item not in learned_types for item in train_y[i]]).to(device)
                    train_y[i].masked_fill_(invalid_mask_label, 0)
                # outputs[:, 0] = 0
                loss, loss_ucl, loss_aug, loss_fd, loss_pd, loss_tlcl = 0, 0, 0, 0, 0, 0
                ce_y = torch.cat(train_y)
                ce_outputs = outputs
                if (args.ucl or args.tlcl) and (stage > 0 or (args.skip_first_cl != "ucl+tlcl" and stage == 0)):                        
                    # _, dpo_feature2 = model(train_x.clone(), train_masks, padded_train_span, span_len)
                    # scl_idx = torch.cat(train_y).nonzero().squeeze(-1)
                    # scl_y = torch.cat(train_y)[scl_idx]
                    # Adj_mat2 = torch.eq(scl_y.unsqueeze(1), scl_y.unsqueeze(1).T).float() - torch.eye(len(scl_y)).to(device)
                    # scl_feat = dpo_feature2[scl_idx, :]
                    # scl_feat = normalize(scl_feat, dim=-1)
                    # logits2 = torch.div(torch.matmul(scl_feat, scl_feat.T), args.cl_temp)
                    # logits_max2, _ = torch.max(logits2, dim=1, keepdim=True)
                    # logits2 = logits2 - logits_max2.detach()
                    # exp_logits2 =  torch.exp(logits2)
                    # denom2 = torch.sum(exp_logits2 * (1 - torch.eye(len(Adj_mat2)).to(device)), dim = -1)
                    # log_prob2 = logits2 - torch.log(denom2)
                    # pos_log_prob2 = torch.sum(Adj_mat2 * log_prob2, dim=-1) / (len(log_prob2) - 1)
                    # loss_scl = -torch.sum(pos_log_prob2)
                    # loss = 0.5 * loss + 0.5 * loss_scl
                    reps = return_dict['reps']
                    bs, hdim = reps.shape
                    aug_repeat_times = args.aug_repeat_times
                    da_x = train_x.clone().repeat((aug_repeat_times, 1))
                    da_y = train_y * aug_repeat_times
                    da_masks = train_masks.repeat((aug_repeat_times, 1))
                    da_span = train_span * aug_repeat_times
                    tk_len = torch.count_nonzero(da_masks, dim=-1) - 2
                    perm = [torch.randperm(item).to(device) + 1 for item in tk_len]
                    if args.cl_aug == "shuffle":
                        for i in range(len(tk_len)):
                            da_span[i] = torch.where(da_span[i].unsqueeze(2) == perm[i].unsqueeze(0).unsqueeze(0))[2].view(-1, 2) + 1
                            da_x[i, 1: 1+tk_len[i]] = da_x[i, perm[i]]
                    elif args.cl_aug =="RTR":
                        rand_ratio = 0.25
                        rand_num = (rand_ratio * tk_len).int()
                        special_ids = [103, 102, 101, 100, 0]
                        all_ids = torch.arange(model.backbone.config.vocab_size).to(device)
                        special_token_mask = torch.ones(model.backbone.config.vocab_size).to(device)
                        special_token_mask[special_ids] = 0
                        all_tokens = all_ids.index_select(0, special_token_mask.nonzero().squeeze())
                        for i in range(len(rand_num)):
                            token_idx = torch.arange(tk_len[i]).to(device) + 1
                            trig_mask = torch.ones(token_idx.shape).to(device)
                            if args.dataset == "ACE":
                                span_pos = da_span[i][da_y[i].nonzero()].view(-1).unique() - 1
                            else:
                                span_pos = da_span[i].view(-1).unique() - 1
                            trig_mask[span_pos] = 0
                            token_idx_ntrig = token_idx.index_select(0, trig_mask.nonzero().squeeze())
                            replace_perm = torch.randperm(token_idx_ntrig.shape.numel())
                            replace_idx = token_idx_ntrig[replace_perm][:rand_num[i]]
                            new_tkn_idx = torch.randperm(len(all_tokens))[:rand_num[i]]
                            da_x[i, replace_idx] = all_tokens[new_tkn_idx].to(device)
                    # if args.dataset == "ACE":
                    #     da_return_dict = model(da_x, da_masks)
                    # else:
                    da_return_dict = model(da_x, da_masks, da_span)
                    da_outputs, da_reps, da_context_feat, da_trig_feat = da_return_dict['outputs'], da_return_dict['reps'], da_return_dict['context_feat'], da_return_dict['trig_feat']
                    
                    if args.ucl:
                        if not ((args.skip_first_cl == "ucl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                            ucl_reps = torch.cat([reps, da_reps])
                            ucl_reps = normalize(ucl_reps, dim=-1)
                            Adj_mask_ucl = torch.zeros(bs * (1 + aug_repeat_times), bs * (1 + aug_repeat_times)).to(device)
                            for i in range(aug_repeat_times):
                                Adj_mask_ucl += torch.eye(bs * (1 + aug_repeat_times)).to(device)
                                Adj_mask_ucl = torch.roll(Adj_mask_ucl, bs, -1)                    
                            loss_ucl = compute_CLLoss(Adj_mask_ucl, ucl_reps, bs * (1 + aug_repeat_times))
                    if args.tlcl:
                        if not ((args.skip_first_cl == "tlcl" or args.skip_first_cl == "ucl+tlcl") and stage == 0):
                            tlcl_feature = torch.cat([trig_feat, da_trig_feat])
                            # tlcl_feature = trig_feat
                            tlcl_feature = normalize(tlcl_feature, dim=-1)
                            tlcl_lbs = torch.cat(train_y + da_y)
                            # tlcl_lbs = torch.cat(train_y)
                            mat_size = tlcl_feature.shape[0]
                            tlcl_lbs_oh = F.one_hot(tlcl_lbs).float()
                            # tlcl_lbs_oh[:, 0] = 0 # whether to compute negative distance
                            Adj_mask_tlcl = torch.matmul(tlcl_lbs_oh, tlcl_lbs_oh.T)
                            Adj_mask_tlcl = Adj_mask_tlcl * (torch.ones(mat_size) - torch.eye(mat_size)).to(device)
                            loss_tlcl = compute_CLLoss(Adj_mask_tlcl, tlcl_feature, mat_size)
                    loss = loss + loss_ucl + loss_tlcl
                    if args.joint_da_loss == "ce" or args.joint_da_loss == "mul":
                        ce_y = torch.cat(train_y + da_y)
                        ce_outputs = torch.cat([outputs, da_outputs])


                
                    # outputs[i].masked_fill_(invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                # if args.dataset == "ACE":
                    
                ce_outputs = ce_outputs[:, learned_types]
                loss_ce = criterion_ce(ce_outputs, ce_y)
                loss = loss + loss_ce
                w = len(prev_learned_types) / len(learned_types)

                if args.rep_aug != "none" and stage > 0:
                    outputs_aug, aug_y = [], []
                    for e_batch in e_loader:
                        exemplar_x, exemplars_y, exemplar_masks, exemplar_span = zip(*e_batch)
                        exemplar_radius = [exemplars.radius[y[0]] for y in exemplars_y]
                        exemplar_x = torch.LongTensor(exemplar_x).to(device)
                        exemplar_masks = torch.LongTensor(exemplar_masks).to(device)
                        exemplars_y = [torch.LongTensor(item).to(device) for item in exemplars_y]
                        exemplar_span = [torch.LongTensor(item).to(device) for item in exemplar_span]            
                        if args.rep_aug == "relative":
                            aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(exemplar_radius)).unsqueeze(-1))
                        else:
                            aug_return_dict = model(exemplar_x, exemplar_masks, exemplar_span, torch.sqrt(torch.stack(list(exemplars.radius.values())).mean()))
                        output_aug = aug_return_dict['outputs_aug']
                        outputs_aug.append(output_aug)
                        aug_y.extend(exemplars_y)
                    outputs_aug = torch.cat(outputs_aug)
                    if args.leave_zero:
                        outputs_aug[:, 0] = 0
                    outputs_aug = outputs_aug[:, learned_types].squeeze(-1)
                    loss_aug = criterion_ce(outputs_aug, torch.cat(aug_y))
                    # loss = loss_ce * w + loss_aug * (1 - w)
                    # loss = loss_ce * (1 - w) + loss_aug * w
                    loss = args.gamma * loss + args.theta * loss_aug
                    

                    

                # if stage > 0 and args.ecl != "none":
                #     _, dpo_feature = model(train_x.clone(), train_masks, padded_train_span, span_len)
                    
                #     # dpo_feature = model.forward_cl(train_x.clone(), train_masks)
                #     ecl_ys, ecl_features = [], []
                #     for e_batch in e_loader:
                #         ecl_x, ecl_y, ecl_masks, ecl_span = zip(*e_batch)
                #         ecl_span_len = [len(item) for item in ecl_span]
                #         ecl_x = torch.LongTensor(ecl_x).to(device)
                #         ecl_masks = torch.LongTensor(ecl_masks).to(device)
                #         ecl_y = [torch.LongTensor(item).to(device) for item in ecl_y]
                #         ecl_span = [torch.LongTensor(item).to(device) for item in ecl_span]            
                #         padded_ecl_span = pad_sequence(ecl_span, batch_first=True, padding_value=-1).to(device)
                #         _, ecl_feature = model(ecl_x, ecl_masks, padded_ecl_span, ecl_span_len)
                #         # ecl_feature = model.forward_cl(ecl_x, ecl_masks)

                #         ecl_features.append(ecl_feature)
                #         ecl_ys.extend(ecl_y)
                #     ecl_ys = torch.cat(ecl_ys)
                #     valid_idx = torch.cat(train_y).nonzero().squeeze(-1)
                #     # feat_idx = [[i] * len(item.nonzero().squeeze(-1)) for (i, item) in enumerate(train_y)]
                #     # s_feat = torch.cat([dpo_feature[i, :] for i in feat_idx])
                #     s_feat = dpo_feature[valid_idx, :]
                #     cl_y = torch.cat(train_y)[valid_idx]
                #     m_index = torch.nonzero(torch.isin(cl_y, ecl_ys)).squeeze(-1)
                #     ecl_index = torch.eq(cl_y.unsqueeze(1), ecl_ys.unsqueeze(1).T).float().argmax(-1)[m_index] # index of exemplars that correspond to the train instance' s label
                #     r_feat = s_feat.clone()
                #     ecl_feat = torch.cat(ecl_features)
                #     r_feat[m_index, :] = ecl_feat[ecl_index, :]
                #     h_feat = normalize(torch.cat((s_feat, r_feat)), dim=-1)
                #     all_y = cl_y.repeat(2)
                #     Adj_mat = torch.eq(all_y.unsqueeze(1), all_y.unsqueeze(1).T).float() - torch.eye(len(all_y)).to(device)
                #     pos_num = torch.sum(Adj_mat, dim=-1)
                #     logits = torch.div(torch.matmul(h_feat, h_feat.T), args.cl_temp)
                #     logits_max, _ = torch.max(logits, dim=1, keepdim=True)
                #     logits = logits - logits_max.detach()
                #     exp_logits =  torch.exp(logits)
                #     denom = torch.sum(exp_logits * (1 - torch.eye(len(Adj_mat)).to(device)), dim = -1)
                #     log_prob = logits - torch.log(denom)
                #     pos_log_prob = torch.sum(Adj_mat * log_prob, dim=-1) / pos_num
                #     loss_scl = -torch.sum(pos_log_prob) / len(pos_log_prob)
                #     loss = 0.5 * loss + 0.5 * loss_scl
                    
                if stage > 0 and args.distill != "none":
                    prev_model.eval()
                    with torch.no_grad():
                        prev_return_dict = prev_model(train_x, train_masks, train_span)
                        prev_outputs, prev_feature = prev_return_dict['outputs'], prev_return_dict['context_feat']

                        if args.joint_da_loss == "dist" or args.joint_da_loss == "mul":
                            outputs = torch.cat([outputs, da_outputs])
                            context_feat = torch.cat([context_feat, da_context_feat])
                            prev_return_dict_cl = prev_model(da_x, da_masks, da_span)
                            prev_outputs_cl, prev_feature_cl = prev_return_dict_cl['outputs'], prev_return_dict_cl['context_feat']
                            prev_outputs, prev_feature = torch.cat([prev_outputs, prev_outputs_cl]), torch.cat([prev_feature, prev_feature_cl])
                    # prev_invalid_mask_op = torch.BoolTensor([item not in prev_learned_types for item in range(args.class_num)]).to(device)
                    prev_valid_mask_op = torch.nonzero(torch.BoolTensor([item in prev_learned_types for item in range(args.class_num + 1)]).to(device))
                    if args.distill == "fd" or args.distill == "mul":
                        prev_feature = normalize(prev_feature.view(-1, prev_feature.shape[-1]), dim=-1)
                        cur_feature = normalize(context_feat.view(-1, prev_feature.shape[-1]), dim=-1)
                        loss_fd = criterion_fd(prev_feature, cur_feature, torch.ones(prev_feature.size(0)).to(device)) # TODO: Don't know whether the code is right
                    else:
                        loss_fd = 0
                    if args.distill == "pd" or args.distill == "mul":
                        T = args.temperature
                        if args.leave_zero:
                            prev_outputs[:, 0] = 0
                        prev_outputs = prev_outputs[:, prev_valid_mask_op].squeeze(-1)
                        cur_outputs = outputs[:, prev_valid_mask_op].squeeze(-1)
                        # prev_outputs[i].masked_fill_(prev_invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                        prev_p = torch.softmax(prev_outputs / T, dim= -1)
                        p = torch.log_softmax(cur_outputs / T, dim = -1)
                        loss_pd = -torch.mean(torch.sum(prev_p * p, dim = -1), dim = 0)
                    else:
                        loss_pd = 0
                    # loss_pd = criterion_pd(torch.cat([item / T for item in outputs]), torch.cat([item / T for item in prev_outputs]))
                    if args.dweight_loss and stage > 0:
                        loss = loss * (1 - w) + (loss_fd + loss_pd) * w
                    else:
                        loss = loss + args.alpha * loss_fd + args.beta * loss_pd
                    # if args.replay and iter_cnt % args.period == 0:
                    #     e_idx = (iter_cnt // args.period - 1) % len(e_loader) 
                    #     ep_x, ep_y, ep_masks, ep_span = zip(*e_loader[e_idx])
                    #     ep_span_len = [len(item) for item in ep_span]
                    #     if np.count_nonzero(ep_span_len) == len(ep_span_len): 
                    #         ep_x = torch.LongTensor(ep_x).to(device)
                    #         ep_masks = torch.LongTensor(ep_masks).to(device)
                    #         ep_y = [torch.LongTensor(item).to(device) for item in ep_y]
                    #         ep_span = [torch.LongTensor(item).to(device) for item in ep_span]                
                    #         padded_ep_span = pad_sequence(ep_span, batch_first=True, padding_value=-1).to(device) 
                    #         e_outputs, e_features = model(ep_x, padded_ep_span, ep_masks, ep_span_len)
                    #         # invalid_mask_op = torch.BoolTensor([item not in learned_types for item in range(args.class_num)]).to(device)
                    #         # not from below's codes
                    #         for i in range(len(ep_y)):
                    #             invalid_mask_e = torch.BoolTensor([item not in learned_types for item in ep_y[i]]).to(device)
                    #             ep_y[i].masked_fill_(invalid_mask_e, 0)
                    #             # outputs[i].masked_fill_(invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                    #         prev_model.eval()
                    #         with torch.no_grad():
                    #             e_prev_outputs, e_prev_features = prev_model(ep_x, padded_ep_span, ep_masks, ep_span_len)
                    #         e_outputs[:, 0] = 0
                    #         e_c_outputs = e_outputs[:, learned_types].squeeze(-1)
                    #         e_loss_ce = criterion_ce(e_c_outputs, torch.cat(ep_y))
                    #         e_prev_features = normalize(e_prev_features, dim=-1)
                    #         e_cur_features = normalize(e_features, dim=-1)
                    #         e_loss_fd = criterion_fd(e_prev_features, e_cur_features, torch.ones(1).to(device)) 
                    #         T = args.temperature
                    #         e_prev_outputs[:, 0] = 0
                    #         e_prev_outputs = e_prev_outputs[:, prev_valid_mask_op].squeeze(-1)
                    #         e_cur_outputs = e_outputs[:, prev_valid_mask_op].squeeze(-1)
                    #                 # prev_outputs[i].masked_fill_(prev_invalid_mask_op, torch.Tensor([float("-inf")]).squeeze(0))
                    #         e_prev_p = torch.softmax(e_prev_outputs / T, dim= -1)
                    #         e_p = torch.log_softmax(e_cur_outputs / T, dim = -1)
                    #         e_loss_pd = -torch.mean(torch.sum(e_prev_p * e_p, dim = -1), dim = 0)
                    #         if args.dweight_loss and stage > 0:
                    #             e_loss = e_loss_ce * (1 - w) + (e_loss_fd + e_loss_pd) * w
                    #         else:
                    #             e_loss = e_loss_ce + args.alpha * e_loss_fd + args.beta * e_loss_pd
                    #             loss = (len(learned_types) * loss + args.e_weight * e_loss) / (len(learned_types) + args.e_weight)
                    

                # if args.amp:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                loss.backward()
                optimizer.step() 
                
            logger.info(f'loss_ce: {loss_ce}')
            logger.info(f'loss_ucl: {loss_ucl}')
            logger.info(f'loss_tlcl: {loss_tlcl}')
            # logger.info(f'loss_ecl: {loss_ecl}')
            logger.info(f'loss_aug: {loss_aug}')
            logger.info(f'loss_fd: {loss_fd}')
            logger.info(f'loss_pd: {loss_pd}')
            logger.info(f'loss_all: {loss}')
            # writer.add_scalar(f'stage{stage}/loss/loss_ce', loss_ce, bt + ep * len(stage_loader))
            # writer.add_scalar(f'stage{stage}/loss/loss_ucl', loss_ucl, bt + ep * len(stage_loader))
            # writer.add_scalar(f'stage{stage}/loss/loss_tlcl', loss_tlcl, bt + ep * len(stage_loader))
            # # writer.add_scalar(f'stage{stage}/loss/loss_ecl', loss_ecl, bt + ep * len(stage_loader))
            # writer.add_scalar(f'stage{stage}/loss/loss_aug', loss_aug, bt + ep * len(stage_loader))
            # writer.add_scalar(f'stage{stage}/loss/loss_fd', loss_fd, bt + ep * len(stage_loader))
            # writer.add_scalar(f'stage{stage}/loss/loss_pd', loss_pd, bt + ep * len(stage_loader))
            # writer.add_scalar(f'stage{stage}/loss/loss_all', loss, bt + ep * len(stage_loader))

            if ((ep + 1) % args.eval_freq == 0 and args.early_stop) or (ep + 1) == args.epochs: # TODO TODO
                # Evaluation process
                logger.info("Evaluation process")
                model.eval()
                with torch.no_grad():
                    if args.single_label:
                        eval_dataset = collect_eval_sldataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item])
                    else:
                        eval_dataset = collect_dataset(args.dataset, args.data_root, 'test', label2idx, None, [i for item in streams for i in item])
                    eval_loader = DataLoader(
                        dataset=eval_dataset,
                        shuffle=False,
                        batch_size=4,
                        collate_fn=lambda x:x)
                    calcs = Calculator()
                    for batch in tqdm(eval_loader):
                        eval_x, eval_y, eval_masks, eval_span = zip(*batch)
                        eval_x = torch.LongTensor(eval_x).to(device)
                        eval_masks = torch.LongTensor(eval_masks).to(device)
                        eval_y = [torch.LongTensor(item).to(device) for item in eval_y]
                        eval_span = [torch.LongTensor(item).to(device) for item in eval_span]  
                        eval_return_dict = model(eval_x, eval_masks, eval_span)
                        eval_outputs = eval_return_dict['outputs']
                        valid_mask_eval_op = torch.BoolTensor([idx in learned_types for idx in range(args.class_num + 1)]).to(device)
                        for i in range(len(eval_y)):
                            invalid_mask_eval_label = torch.BoolTensor([item not in learned_types for item in eval_y[i]]).to(device)
                            eval_y[i].masked_fill_(invalid_mask_eval_label, 0)
                        if args.leave_zero:
                            eval_outputs[:, 0] = 0
                        eval_outputs = eval_outputs[:, valid_mask_eval_op].squeeze(-1)
                        calcs.extend(eval_outputs.argmax(-1), torch.cat(eval_y))
                    bc, (precision, recall, micro_F1) = calcs.by_class(learned_types)
                    if args.log:
                        writer.add_scalar(f'score/epoch/marco_F1', micro_F1,  ep + 1 + args.epochs * stage)
                    if args.log and (ep + 1) == args.epochs:
                        writer.add_scalar(f'score/stage/marco_F1', micro_F1, stage)
                    logger.info(f'marco F1 {micro_F1}')
                    dev_scores_ls.append(micro_F1)
                    logger.info(f"Dev scores list: {dev_scores_ls}")
                    logger.info(f"bc:{bc}")
                    if args.early_stop:
                        if dev_score is None or dev_score < micro_F1:
                            no_better = 0
                            dev_score = micro_F1
                            torch.save(model.state_dict(), e_pth)
                        else:
                            no_better += 1
                            logger.info(f'No better: {no_better}/{args.patience}')
                        if no_better >= args.patience:
                            logger.info("Early stopping with dev_score: " + str(dev_score))
                            if args.log:
                                writer.add_scalar(f'score/stage/marco_F1', micro_F1, stage)
                            break

        for tp in streams_indexed[stage]:
            if not tp == 0:
                labels.pop(labels.index(tp))
        save_stage = stage
        if args.save_dir and local_rank == 0:
            state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'stage':stage + 1, 
                            'labels':labels, 'learned_types':learned_types, 'prev_learned_types':prev_learned_types}
            save_pth = os.path.join(args.save_dir, "perm" + str(args.perm_id))
            save_name = f"stage_{save_stage}_{cur_time}.pth"
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            logger.info(f'state_dict saved to: {os.path.join(save_pth, save_name)}')
            torch.save(state, os.path.join(save_pth, save_name))
            os.remove(e_pth)





if __name__ == "__main__":
    args = parse_arguments()
    if args.parallel == 'DDP':
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(train,
            args=(args, ),
            nprocs=args.world_size,
            join=True)
    else:
        train(0, args)
        
