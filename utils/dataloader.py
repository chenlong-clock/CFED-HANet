from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from configs import parse_arguments
from utils.tools import collect_from_json

import numpy as np
args = parse_arguments()



class MAVEN_Dataset(Dataset):
    def __init__(self, tokens, labels, masks, spans) -> None:
        super(Dataset).__init__()
        self.tokens = tokens
        self.masks = masks
        self.labels = labels
        self.spans = spans
        # self.requires_cl = [0 for _ in range(len(spans))]
        # self.labels = []
        # for stream in streams:
        #     for lb in stream:
        #         if not lb in self.label2idx:
        #             self.label2idx[lb] = len(self.label2idx)
        # for label_ls in labels:
        #     self.labels.append([self.label2idx[label]  for label in label_ls])
    def __getitem__(self, index):
        return [self.tokens[index], self.labels[index], self.masks[index], self.spans[index]]
    def __len__(self):
        return len(self.labels)
    def extend(self, tokens, labels, masks, spans):
        self.tokens.extend(tokens)
        self.labels.extend(labels)
        self.masks.extend(masks)
        self.spans.extend(spans)
        # self.requires_cl.extend(requires_cl)
    # def collate_fn(self, batch):
    #     batch = pad_sequence([torch.LongTensor(item) for item in batch[2]])
    #     pass

def collect_dataset(dataset, root, split, label2idx, stage_id, labels):
    if split == 'train':
        data = [instance for t in collect_from_json(dataset, root, split)[stage_id] for instance in t]
    else:
        data = collect_from_json(dataset, root, split)
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for dt in tqdm(data):
        # pop useless properties
        if 'mention_id' in dt.keys():
            dt.pop('mention_id')
        if 'sentence_id' in dt.keys():    
            dt.pop('sentence_id')
        # if split == 'train':
        add_label = []
        add_span = []
        new_t = {}
        for i in range(len(dt['label'])):
            if dt['label'][i] in labels or dt['label'][i] == 0: # if the label of instance is in the query
                add_label.append(dt['label'][i]) # append the instance and the label
                add_span.append(dt['span'][i]) # the same as label
        if len(add_label) != 0:
            token = dt['piece_ids']
            new_t['label'] = add_label
            valid_span = add_span
            valid_label = [label2idx[item] if item in label2idx else 0 for item in add_label]
        # else:
        #     token = dt['piece_ids']
        #     valid_span = dt['span'].copy()
        #     valid_label = [label2idx[item] if item in label2idx else 0 for item in dt['label']]
            # max_seqlen = 90
        max_seqlen = args.max_seqlen # 344, 249, 230, 186, 167
        if len(token) >= max_seqlen + 2:
            token_sep = token[-1]
            token = token[:max_seqlen + 1] + [token_sep]
            invalid_span = np.unique(np.nonzero(np.asarray(valid_span) > max_seqlen)[0])
            invalid_span = invalid_span[::-1]
            for invalid_idx in invalid_span:
                valid_span.pop(invalid_idx)
                valid_label.pop(invalid_idx)
        if len(token) < max_seqlen + 2:
            token = token + [0] * (max_seqlen + 2 - len(token))
        token_mask = [1 if tkn != 0 else 0 for tkn in token]
            # span_mask = []
            # for i in range(len(token)):
            #     span_mask.append([0, 0])
            # for item in valid_span:
            #     for i in range(len(item)):
            #         span_mask[item[i]][i] = 1
        data_tokens.append(token)
        data_labels.append(valid_label)
        data_masks.append(token_mask)
        data_spans.append(valid_span)
            # data_spans.append(valid_span)
    if args.my_test:
        return MAVEN_Dataset(data_tokens[:100], data_labels[:100], data_masks[:100], data_spans[:100]) # TODO: deprecated, used for debugging, not for test!
    else:
        return MAVEN_Dataset(data_tokens, data_labels, data_masks, data_spans)


def collect_exemplar_dataset(dataset, root, split, label2idx, stage_id, labels):
    data = [[instance for instance in t] for t in collect_from_json(dataset, root, split)[stage_id]]
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for idx, task_data in enumerate(tqdm(data)):
        for dt in task_data:
            # pop useless properties
            if 'mention_id' in dt.keys():
                dt.pop('mention_id')
            if 'sentence_id' in dt.keys():    
                dt.pop('sentence_id')
            # if split == 'train':
            add_label = []
            add_span = []
            new_t = {}
            for i in range(len(dt['label'])):
                if dt['label'][i] == labels[idx]: 
                    add_label.append(dt['label'][i]) 
                    add_span.append(dt['span'][i])
            if len(add_label) != 0:
                token = dt['piece_ids']
                new_t['label'] = add_label
                valid_span = add_span
                valid_label = [label2idx[item] if item in label2idx else 0 for item in add_label]
            # else:
            #     token = dt['piece_ids']
            #     valid_span = dt['span'].copy()
            #     valid_label = [label2idx[item] if item in label2idx else 0 for item in dt['label']]
                # max_seqlen = 90
            max_seqlen = args.max_seqlen # 344, 249, 230, 186, 167
            if len(token) >= max_seqlen + 2:
                token_sep = token[-1]
                token = token[:max_seqlen + 1] + [token_sep]
                invalid_span = np.unique(np.nonzero(np.asarray(valid_span) > max_seqlen)[0])
                invalid_span = invalid_span[::-1]
                for invalid_idx in invalid_span:
                    valid_span.pop(invalid_idx)
                    valid_label.pop(invalid_idx)
            if len(token) < max_seqlen + 2:
                token = token + [0] * (max_seqlen + 2 - len(token))
            token_mask = [1 if tkn != 0 else 0 for tkn in token]
                # span_mask = []
                # for i in range(len(token)):
                #     span_mask.append([0, 0])
                # for item in valid_span:
                #     for i in range(len(item)):
                #         span_mask[item[i]][i] = 1
            data_tokens.append(token)
            data_labels.append(valid_label)
            data_masks.append(token_mask)
            data_spans.append(valid_span)
    return MAVEN_Dataset(data_tokens, data_labels, data_masks, data_spans)
                

def collect_sldataset(dataset, root, split, label2idx, stage_id, labels):
    data = [[instance for instance in t] for t in collect_from_json(dataset, root, split)[stage_id]]
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for idx, task_data in enumerate(tqdm(data)):
        for dt in task_data:
            # pop useless properties
            if 'mention_id' in dt.keys():
                dt.pop('mention_id')
            if 'sentence_id' in dt.keys():    
                dt.pop('sentence_id')
            # if split == 'train':
            add_label = []
            add_span = []
            new_t = {}
            for i in range(len(dt['label'])):
                if dt['label'][i] == labels[idx] or dt['label'][i] == 0: 
                    add_label.append(dt['label'][i]) 
                    add_span.append(dt['span'][i])
            if len(add_label) != 0:
                token = dt['piece_ids']
                new_t['label'] = add_label
                valid_span = add_span
                valid_label = [label2idx[item] if item in label2idx else 0 for item in add_label]
            # else:
            #     token = dt['piece_ids']
            #     valid_span = dt['span'].copy()
            #     valid_label = [label2idx[item] if item in label2idx else 0 for item in dt['label']]
                # max_seqlen = 90
            max_seqlen = args.max_seqlen # 344, 249, 230, 186, 167
            if len(token) >= max_seqlen + 2:
                token_sep = token[-1]
                token = token[:max_seqlen + 1] + [token_sep]
                invalid_span = np.unique(np.nonzero(np.asarray(valid_span) > max_seqlen)[0])
                invalid_span = invalid_span[::-1]
                for invalid_idx in invalid_span:
                    valid_span.pop(invalid_idx)
                    valid_label.pop(invalid_idx)
            if len(token) < max_seqlen + 2:
                token = token + [0] * (max_seqlen + 2 - len(token))
            token_mask = [1 if tkn != 0 else 0 for tkn in token]

                # span_mask = []
                # for i in range(len(token)):
                #     span_mask.append([0, 0])
                # for item in valid_span:
                #     for i in range(len(item)):
                #         span_mask[item[i]][i] = 1
            data_tokens.append(token)
            data_labels.append(valid_label)
            data_masks.append(token_mask)
            data_spans.append(valid_span)
    if args.my_test:
        return MAVEN_Dataset(data_tokens[:100], data_labels[:100], data_masks[:100], data_spans[:100]) # TODO:test use
    else:
        return MAVEN_Dataset(data_tokens, data_labels, data_masks, data_spans)

def collect_eval_sldataset(dataset, root, split, label2idx, stage_id, labels):
    data = collect_from_json(dataset, root, split)
    data_tokens, data_labels, data_masks, data_spans = [], [], [], []
    for dt in tqdm(data):
        # pop useless properties
        if 'mention_id' in dt.keys():
            dt.pop('mention_id')
        if 'sentence_id' in dt.keys():    
            dt.pop('sentence_id')
        # if split == 'train':
        add_label = []
        add_span = []
        new_t = {}
        for i in range(len(dt['label'])):
            if dt['label'][i] in labels or dt['label'][i] == 0: # if the label of instance is in the query
                add_label.append(dt['label'][i]) # append the instance and the label
                add_span.append(dt['span'][i]) # the same as label
        if len(add_label) != 0:
            token = dt['piece_ids']
            new_t['label'] = add_label
            valid_span = add_span
            valid_label = [label2idx[item] if item in label2idx else 0 for item in add_label]
        # else:
        #     token = dt['piece_ids']
        #     valid_span = dt['span'].copy()
        #     valid_label = [label2idx[item] if item in label2idx else 0 for item in dt['label']]
            # max_seqlen = 90
        max_seqlen = args.max_seqlen # 344, 249, 230, 186, 167
        if len(token) > max_seqlen + 2:
            token_sep = token[-1]
            token = token[:max_seqlen + 1] + [token_sep]
            invalid_span = np.unique(np.nonzero(np.asarray(valid_span) > max_seqlen)[0])
            invalid_span = invalid_span[::-1]
            for invalid_idx in invalid_span:
                valid_span.pop(invalid_idx)
                valid_label.pop(invalid_idx)
        if len(token) < max_seqlen + 2:
            token = token + [0] * (max_seqlen + 2 - len(token))
        token_mask = [1 if tkn != 0 else 0 for tkn in token]
            # span_mask = []
            # for i in range(len(token)):
            #     span_mask.append([0, 0])
            # for item in valid_span:
            #     for i in range(len(item)):
            #         span_mask[item[i]][i] = 1
        data_tokens.append(token)
        data_labels.append(valid_label)
        data_masks.append(token_mask)
        data_spans.append(valid_span)
            # data_spans.append(valid_span)
    if args.my_test:
        return MAVEN_Dataset(data_tokens[:100], data_labels[:100], data_masks[:100], data_spans[:100]) # TODO:test use
    else:
        return MAVEN_Dataset(data_tokens, data_labels, data_masks, data_spans)


# def collect_fewshot_dataset(dataset, root, split, labels, label2idx):
#         # collect instances that have label in the list 'labels' 
#         data = collect_from_json(dataset, root, split)
#         data_tokens, data_labels, data_masks, data_spans = [], [], [], []
#         data_ls = []
#         for t in data:
#             task_ls = []
#             for val in t.values():
#                 for item in val:
#                     for lb in item['label']:
#                         if lb != 0 and lb in labels:
#                             task_ls.append(item)
#                             break
#             data_ls.append(task_ls)
#         return data_ls    

# def get_data_loaders(dataset, root, streams, batch_size=1):
#     train_loaders = []
#     train_ecn_loaders = []
#     data_train = collect_from_json(dataset, root, 'train') # load dataset
#     data_dev = collect_from_json(dataset, root, 'dev')
#     all_labels = []
#     all_labels = list(set([t for stream in streams for t in stream if t not in all_labels]))
#     labels = all_labels.copy()
#     for i, stream in enumerate(streams): # get data from labels for each stream
#         print(f'loading train instances for stage {i}') 
#         stream_instances_loader = DataLoader(
#             dataset=collect_dataset('MAVEN', root, 'train', labels),
#             batch_size=batch_size,
#             shuffle=True,
#             drop_last=False)
#         print(f"stage {i}: instances num {len(stream_instances_loader)}")
#         train_loaders.append(stream_instances_loader)
#         print(f'loading train instances excluding none for stage {i}') 
#         exclude_none_labels = [t for t in stream if t != 0]
#         exclude_none_loader = DataLoader(
#             dataset=collect_dataset('MAVEN', root, 'train', exclude_none_labels),
#             batch_size=batch_size,
#             shuffle=True,
#             drop_last=False)
#         train_ecn_loaders.append(exclude_none_loader)
#         print(f"stage {i}: instances num {len(exclude_none_loader)}")
#         for tp in stream:
#             if not tp == 0:
#                 labels.pop(labels.index(tp))
#     print(f'loading dev instances')
#     dev_loaders = [DataLoader(
#             dataset=collect_dataset(data_dev, all_labels),
#             batch_size=batch_size,
#             shuffle=True,
#             drop_last=False)]
#     print(f'loading test instances')

#     data_test = collect_from_json(dataset, root, 'test')
#     test_loaders = [DataLoader(
#             dataset=collect_dataset(data_test, all_labels),
#             batch_size=batch_size,
#             shuffle=True,
#             drop_last=False)]
#     loaders_dict = {'train': train_loaders, 'train-exclude-none':train_ecn_loaders, 
#                     'dev': dev_loaders, 'test': test_loaders}
#     return loaders_dict


    
# def test():
#     streams = collect_from_json('MAVEN', './data', 'streams')
#     get_data_loaders('MAVEN', './data', streams)
# if __name__ == "__main__":
#     test()
