import json
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import SmoothL1Loss, MSELoss
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.util import StandardScaler2
from models.main_model import Cl_TTE
import ast
highway = {'<PAD>': 0, 'unclassified': 1, 'busway': 2, 'crossing': 3, 'living_street': 4, 'motorway': 5, 'motorway_link': 6, 'primary': 7, 'primary_link': 8, 'residential': 9, 'road': 10, 'secondary': 11, 'secondary_link': 12, 'tertiary': 13, 'tertiary_link': 14, 'trunk': 15, 'trunk_link': 16}
def augment_segments(seg,
                     p_highway=0.2,
                     p_poi=0.35,
                     p_seg=0.12):

    seg_aug = seg * 1.0 # (T, F)

    # --- Highway dropout ---
    highway = seg_aug[:, :2]
    mask_hw = np.random.rand(*highway.shape) < p_highway
    highway[mask_hw] = 1  # unclassified
    seg_aug[:, :2] = highway

    # --- POI dropout ---
    poi = seg_aug[:, 8:]
    mask_poi = np.random.rand(*poi.shape) < p_poi
    poi = poi * (~mask_poi)
    seg_aug[:, 8:] = poi

    # --- Segment dropout (feature masking, NOT removal) ---
    T = seg_aug.shape[0]
    seg_mask = np.random.rand(T) < p_seg

    # prevent full collapse
    if seg_mask.all():
        seg_mask[np.random.randint(T)] = False
        
    seg_aug[seg_mask, 0:2] = 1  # highway → unclassified
    seg_aug[seg_mask, 4:8] *= 0.3  # GPS
    seg_aug[seg_mask, 8:] *= 0.3  # POI

    return seg_aug
def preprocess_edgeinfo(edgeinfo):
    new_edgeinfo = {}

    for k, info in edgeinfo.items():
        hw_ids = parse_highway_tags(info[0])  # run ONCE

        new_edgeinfo[k] = [
            hw_ids,      # already parsed
            info[1],
            info[2],
            info[3],
            *info[4:]
        ]

    return new_edgeinfo
def parse_highway_tags(raw_val, max_tags=2):
    UNCLASSIFIED_ID = 1

    if isinstance(raw_val, list):
        tags = raw_val
    elif isinstance(raw_val, str) and raw_val.startswith("["):
        try:
            tags = raw_val.strip("[]").replace("'", "").split(",")
        except:
            tags = [raw_val]
    else:
        tags = [raw_val]

    ids = [highway.get(t.strip(), UNCLASSIFIED_ID) for t in tags[:max_tags]]

    if len(ids) < max_tags:
        ids += [0] * (max_tags - len(ids))

    return ids

node_type = {'turning_circle':1, 'traffic_signals':2, 'crossing':3, 'motorway_junction':4, "mini_roundabout":5}

def collate_func(data, args, info_all):
    edgeinfo, nodeinfo, scaler, scaler2 = info_all

    time = torch.Tensor([d[-1] for d in data])
    linkids = [np.asarray(l[1]) for l in data]
    dateinfo = []
    inds = []
    n_poi_groups = args.data_config['n_poi_groups'] 

    for l in data:
        wday = int(l[2])
        doy_norm = (float(l[3]) / 365.0) * 2 * np.pi
        minute_norm = (float(l[4]) / 1440.0) * 2 * np.pi
        dateinfo.append([wday, doy_norm, minute_norm])
        inds.append(l[0])
    
    lens = np.array([len(k) for k in linkids])
    max_seq_len = lens.max()
    
    def get_infos(xs):
        L = len(xs)
        feat_dim = 8 + len(edgeinfo[xs[0]][4:])
        
        seg = np.zeros((L, feat_dim), dtype=np.float32)
        
        for i, x in enumerate(xs):
            info = edgeinfo[x]
            
            seg[i, :2] = info[0]
            seg[i,2] = info[1]
            
            n1, n2 = info[2], info[3]

            if n1 in nodeinfo and n2 in nodeinfo:
                seg[i, 4:8] = [
                    nodeinfo[info[2]][0], nodeinfo[info[2]][1],
                    nodeinfo[info[3]][0], nodeinfo[info[3]][1]
                ]
            else:
                seg[i, 4:8] = 0.0
            
            seg[i, 8:] = info[4:]
        
        lengths = seg[:, 2]
        cum = np.cumsum(lengths)
        seg[:, 3] = np.concatenate([[0], cum[:-1]])
        
        return seg
    
    total_len = sum(lens)
    feature_dim = 8 + len(edgeinfo[linkids[0][0]][4:])
    all_segments = np.zeros((total_len, feature_dim), dtype=np.float32)

    ptr = 0
    for b in linkids:
        seg = get_infos(b)
        L = len(seg)
        
        all_segments[ptr:ptr+L] = seg
        ptr += L
        
    # Scale: Length (idx 2), CumLen (idx 3)
    all_segments[:, 2:4] = scaler.transform(all_segments[:, 2:4])
    # Scale: GPS (idx 4 to 7)
    all_segments[:, 4:8] = scaler2.transform(all_segments[:, 4:8])

    all_segments = np.nan_to_num(all_segments, 0.0)
    all_segments[:, 8:] = np.maximum(all_segments[:, 8:], 0)
    
    # Shape: [Batch, Max_Seq, 8] 
    # Features: [HighwayID1, HighwayID2, Len, CumLen, Lat1, Lon1, Lat2, Lon2]
    feature_dim = all_segments.shape[1]
    padded_clean = np.zeros((len(data), max_seq_len, feature_dim), dtype=np.float32)
    padded_aug   = np.zeros_like(padded_clean)
    
    curr_idx = 0
    for i, l in enumerate(lens):
        if curr_idx + l > len(all_segments):
            print(f"Overflow or mismatch: curr_idx={curr_idx}, l={l}, total={len(all_segments)}")
        seg = all_segments[curr_idx : curr_idx + l]
        
        if seg.shape[0] != l:
            print(f"Mismatch at batch {i}: expected {l}, got {seg.shape[0]}")
            
        padded_clean[i, :l] = seg
        
        # augmented view
        seg_aug = augment_segments(seg)
        padded_aug[i, :l] = seg_aug

        curr_idx += l
    
    return {
        'links_clean': torch.from_numpy(padded_clean),
        'links_aug': torch.from_numpy(padded_aug),
        'dateinfo': torch.from_numpy(np.asarray(dateinfo, dtype=np.float32)),
        'lens': torch.LongTensor(lens), 
        'inds': inds, 
    }, time

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        if isinstance(dataset[0], dict):
            self.lengths = [len(d['lats']) for d in dataset]
        elif isinstance(dataset[0][1], list):
            self.lengths = [len(d[1]) for d in dataset]
        else:
            self.lengths = [d[0]['lens'] for d in dataset]
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batcha
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def load_datadoct_pre(args):
    global info_all
    
    abspath = os.path.join(os.path.dirname(__file__), "data_config.json")
    with open(abspath) as file:
        data_config = json.load(file)[args.dataset]
        args.data_config = data_config
    
    with open(os.path.join(args.absPath,args.data_config['edges_dir']), 'rb') as f:
        edgeinfo = pickle.load(f)
    new_edgeinfo = preprocess_edgeinfo(edgeinfo)
    
    with open(os.path.join(args.absPath,args.data_config['nodes_dir']), 'rb') as f:
        nodeinfo = pickle.load(f)
    
    if "porto" in args.dataset:
        scaler = StandardScaler()
        scaler.fit([[0, 0]])
        scaler.mean_ = [107.497195, 3010.37456]
        scaler.scale_ = [131.102877, 2750.78118]
        scaler2 = StandardScaler()
        scaler2.fit([[0, 0, 0, 0]])
        scaler2.mean_ = [-8.62247695, 41.15923239, -8.62256569, 41.15929004]
        scaler2.scale_ = [0.02520552, 0.01236445, 0.02526226, 0.01242564]

        
    elif "chengdu" in args.dataset:
        scaler = StandardScaler()
        scaler.fit([[0,0]])
        scaler.mean_ = [188.285260, 3969.52982]
        scaler.scale_ = [206.040346, 3658.76429]
        scaler2 = StandardScaler()
        scaler2.fit([[0,0,0,0]])
        scaler2.mean_ = [104.06379941,  30.65844312, 104.06381633,  30.65845601]
        scaler2.scale_ = [0.03480474, 0.02717924, 0.03484908, 0.02719959]
        
    elif "hcm" in args.dataset:
        scaler = StandardScaler()
        scaler.fit([[0,0]])
        scaler.mean_ = [94.526, 5133.815]
        scaler.scale_ = [110.723, 3745.034]
        
        scaler2 = StandardScaler()
        scaler2.fit([[0,0,0,0]])

        scaler2.mean_ = [106.665882,  10.781017, 106.665884,  10.781012]
        scaler2.scale_ = [0.022315, 0.025133, 0.022316, 0.025130]
    else:
        ValueError("Wrong Dataset Name")

    info_all = [new_edgeinfo, nodeinfo, scaler, scaler2]


class Datadict(Dataset):
    def __init__(self, inputs):
        self.content = inputs

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)
def load_test_datadict(args):
    tdata = np.load(os.path.join(args.absPath,args.data_config['data_dir'],'test.npy'), allow_pickle=True)
    test_loader = DataLoader(Datadict(tdata), batch_size=args.batch_size,
                                        collate_fn=lambda x: collate_func(x, args, info_all),
                                        pin_memory=True, shuffle=False)
    
    return test_loader, StandardScaler2(mean=args.data_config['time_mean'], std=args.data_config['time_std'])
def load_datadict(args):
    data = {}
    loader = {}
    if args.mode == 'test':
        phases = ['test']
    else:
        phases = ['train', 'val']

    for phase in phases:
        tdata = np.load(os.path.join(args.absPath,args.data_config['data_dir'], phase + '.npy'), allow_pickle=True)
        data[phase] = tdata

        if phase == 'train':
            loader[phase] = DataLoader(Datadict(data[phase]), batch_sampler=BatchSampler(data[phase], args.batch_size),
                                        collate_fn=lambda x: collate_func(x, args, info_all),
                                        pin_memory=True,num_workers=2)
        else:
            
            loader[phase] = DataLoader(Datadict(data[phase]), batch_size=args.batch_size,
                                        collate_fn=lambda x: collate_func(x, args, info_all),
                                        shuffle=False, pin_memory=True,num_workers=2)
    return loader.copy(), StandardScaler2(mean=args.data_config['time_mean'], std=args.data_config['time_std'])


def create_model(args):
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath) as file:
        model_config = json.load(file)[args.model]
        
    args.model_config = model_config
    args.model_config['n_poi_groups'] = args.data_config["n_poi_groups"]
    args.model_config['r_percentile'] = args.data_config["r_percentile"]
    
    return Cl_TTE(**model_config)
        

def create_main_loss(loss_eta,loss_cl, args):
    beta = args.beta
    
    # scale = (loss_eta.detach() / (loss_cl.detach() + 1e-6))
    scale = 1 / (loss_cl / loss_eta + 1e-4).detach()
    # loss_cl_scaled = loss_cl * scale.clamp(0.1, 10.0)   
    loss_cl_scaled = loss_cl * scale.clamp(0.01, 10.0)
    
    return beta * loss_eta + (1 - beta) * loss_cl_scaled
  
def create_loss(args):
    if args.loss == 'rmse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            rmse = torch.sqrt(torch.mean(torch.pow(preds - labels, 2)))
            return rmse
    elif args.loss == 'mse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            # mse = torch.mean(torch.pow(preds - labels, 2))
            mse = MSELoss(reduction='mean').forward(preds.view(-1), labels)
            return mse
    elif args.loss == 'mape':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels) / (labels + 0.1))
            return mape
    elif args.loss == 'mae':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels))
            return mape
    elif args.loss == 'smoothL1':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            preds = torch.squeeze(preds, 1)
            smoothL1 = SmoothL1Loss(reduction='mean', beta = args.loss_val).forward(preds, labels)
            return smoothL1

    else:
        raise ValueError("Unknown loss function.")
    return loss

