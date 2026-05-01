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
SPEED_BUCKETS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120]
# 0 = unknown

def snap_to_bucket(speed: int) -> int:
    """Snap a speed value to the nearest bucket index."""
    if speed < 10:
        return 0   # treat as unknown/invalid
    thresholds = SPEED_BUCKETS[1:]   # exclude 0 (unknown)
    for i in range(len(thresholds) - 1):
        mid = (thresholds[i] + thresholds[i+1]) / 2
        if speed <= mid:
            return i + 1   # +1 because index 0 is unknown
    return len(thresholds)   # max bucket

def parse_speed_value(val: str) -> int:
    """Parse a single speed string to integer km/h."""
    val = val.strip().lower()
    val = val.replace('km/h', '').replace('mph', '').strip()
    try:
        speed = int(float(val))
        if 'mph' in str(val):
            speed = int(speed * 1.609)
        return speed
    except ValueError:
        return 0

def parse_maxspeed(raw_val) -> int:
    """
    Parse OSM maxspeed tag to bucket index.
    For list values, takes the minimum valid speed.
    Returns 0 for unknown/invalid.
    """
    if raw_val is None or raw_val in ('', 'unknown', 'none', 'signals'):
        return 0

    if raw_val in ('walk', 'living_street'):
        return 1   # ~10 km/h bucket

    raw_str = str(raw_val).strip()

    # handle list stored as string: "['50', '40']"
    if raw_str.startswith('['):
        try:
            # strip brackets and quotes, split by comma
            inner = raw_str.strip("[]").replace("'", "").replace('"', '')
            parts = [p.strip() for p in inner.split(',')]
            speeds = [parse_speed_value(p) for p in parts]
            speeds = [s for s in speeds if s >= 10]   # filter invalid
            if not speeds:
                return 0
            speed = min(speeds)   # take most restrictive
        except Exception:
            return 0
    else:
        speed = parse_speed_value(raw_str)
        if speed < 10:
            return 0

    return snap_to_bucket(speed)

LANE_BUCKETS = {
    0: 0,   # unknown
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,   # cap at 6, anything above is rare and treated as 6
}
N_LANE_BUCKETS = 7   # indices 0-6

def parse_lane_value(val: str) -> int:
    """Parse a single lane string to integer."""
    try:
        return max(1, min(6, int(float(val.strip()))))
    except (ValueError, TypeError):
        return 0

def parse_lanes(raw_val) -> int:
    """
    Parse OSM lanes tag to bucket index.
    For list values, takes the minimum (most restrictive).
    Returns 0 for unknown/invalid.
    """
    if raw_val is None or str(raw_val).strip() in ('', 'unknown', 'none'):
        return 0

    raw_str = str(raw_val).strip()

    # handle list stored as string: "['2', '1']"
    if raw_str.startswith('['):
        try:
            inner  = raw_str.strip("[]").replace("'", "").replace('"', '')
            parts  = [p.strip() for p in inner.split(',')]
            lanes  = [parse_lane_value(p) for p in parts if p]
            lanes  = [l for l in lanes if l > 0]
            return min(lanes) if lanes else 0
        except Exception:
            return 0

    # handle semicolon format: '1;2'
    if ';' in raw_str:
        parts = [p.strip() for p in raw_str.split(';')]
        lanes = [parse_lane_value(p) for p in parts if p]
        lanes = [l for l in lanes if l > 0]
        return min(lanes) if lanes else 0

    return parse_lane_value(raw_str)

def augment_segments(seg,
                     p_highway=0.1,
                     p_poi=0.15,
                     p_seg=0.2,
                     max_percent_merge= 0.2
                     ):

    seg_aug = seg.copy() # (T, F)
    T, F = seg_aug.shape
    
    if T > 5: # Only merge edges if the trajectory is reasonably long
        total_len = seg_aug[:, 2].sum()
        budget = total_len * max_percent_merge
        absorbed = 0.0
        i = 0
        while i < seg_aug.shape[0] - 1 and absorbed < budget:
            if np.random.rand() < 0.5:
                len_a = seg_aug[i, 2]
                len_b = seg_aug[i + 1, 2]

                if absorbed + len_b > budget:
                    i += 1
                    continue

                total = len_a + len_b
                seg_aug[i, 2] = total

                spd_a, spd_b = seg_aug[i, 4], seg_aug[i+1, 4]
                if spd_a > 0 and spd_b > 0:
                    seg_aug[i, 4] = round((spd_a * len_a + spd_b * len_b) / total)
                elif spd_a > 0:
                    seg_aug[i, 4] = spd_a
                else:
                    seg_aug[i, 4] = spd_b

                ln_a, ln_b = seg_aug[i, 5], seg_aug[i+1, 5]
                if ln_a > 0 and ln_b > 0:
                    seg_aug[i, 5] = round((ln_a * len_a + ln_b * len_b) / total)
                elif ln_a > 0:
                    seg_aug[i, 5] = ln_a
                else:
                    seg_aug[i, 5] = ln_b

                if len_b > len_a:
                    seg_aug[i, :2] = seg_aug[i+1, :2]

                seg_aug[i, 6:] += seg_aug[i+1, 6:]
                seg_aug = np.delete(seg_aug, i+1, axis=0)
                absorbed += len_b
            else:
                i += 1
                
    T = seg_aug.shape[0]

    # --- Highway dropout ---
    highway = seg_aug[:, :2]
    mask_hw = np.random.rand(*highway.shape) < p_highway
    highway[mask_hw] = 1  # unclassified
    seg_aug[:, :2] = highway

    # --- POI dropout ---
    poi = seg_aug[:, 6:]
    mask_poi_rows = np.random.rand(T, 1) < p_poi
    poi = poi * (~mask_poi_rows)
    seg_aug[:, 6:] = poi

    # --- Segment dropout (feature masking, NOT removal) ---
    seg_mask = np.random.rand(T) < p_seg

    # prevent full collapse
    if seg_mask.all():
        seg_mask[np.random.randint(T)] = False
        
    seg_aug[seg_mask, 0:2] = 1  # highway → unclassified
    seg_aug[seg_mask, 4] = 0  # speed bucket → unknown
    seg_aug[seg_mask, 5] = 0  # lane bucket → unknown
    seg_aug[seg_mask, 6:] = 0  # pois → no pois
    
    noise = np.random.normal(1.0, 0.03, size=seg_aug[:, 2].shape)  # ±3% multiplicative
    seg_aug[:, 2] *= noise
    seg_aug[:, 2] = np.clip(seg_aug[:, 2], a_min=1.0, a_max=None)  # ensure length is positive
    
    cum = np.cumsum(seg_aug[:, 2])
    seg_aug[:, 3] = np.concatenate([[0], cum[:-1]])
    
    return seg_aug, T

def preprocess_edgeinfo(edgeinfo,args):
    new_edgeinfo = {}

    # edge_neighbors = build_edge_adjacency(edgeinfo)
    # deg_in, deg_out = compute_node_degree(edgeinfo)
    
    for k, info in edgeinfo.items():
        hw_ids = parse_highway_tags(info[0])
        speed_bucket = parse_maxspeed(info[4 + args.data_config['n_poi_groups']])
        lane_bucket = parse_lanes(info[4 + args.data_config['n_poi_groups'] + 1])
        poi_self = np.array(info[4:4 + args.data_config['n_poi_groups']], dtype=np.float32)
        
        new_edgeinfo[k] = [
            hw_ids,      # already parsed
            info[1],
            speed_bucket,  # parsed speed bucket
            lane_bucket,         # parsed lanes
            *poi_self,
        ] # 1 + 1 + n_poi_groups = 1 + 1 + n_poi_groups features per edge

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

def collate_func(data, args, info_all):
    edgeinfo, scaler = info_all

    time = torch.Tensor([d[-1] for d in data])
    linkids = [np.asarray(l[1]) for l in data]
    dateinfo = []
    inds = []

    for l in data:
        wday = int(l[2])
        doy_raw = float(l[3])        # 1-365
        minute_raw = float(l[4])
        dateinfo.append([wday, doy_raw, minute_raw])
        inds.append(l[0])
    
    lens = np.array([len(k) for k in linkids])
    max_seq_len = lens.max()
    
    feature_dim = 4 + 2 + args.data_config['n_poi_groups'] # highway(2), length(1), cum_length(1), pois(n_poi_groups)
    
    def get_infos(xs):
        L = len(xs)
        
        seg = np.zeros((L, feature_dim), dtype=np.float32)
        
        for i, x in enumerate(xs):
            info = edgeinfo[x]
            
            seg[i, :2] = info[0]
            seg[i,2] = info[1] # length
            seg[i, 4] = info[2] # speed_bucket
            seg[i, 5] = info[3] # lane_bucket
            
            seg[i, 6:6+args.data_config['n_poi_groups']] = info[4:4+args.data_config['n_poi_groups']]
            
        lengths = seg[:, 2]
        cum = np.cumsum(lengths)
        seg[:, 3] = np.concatenate([[0], cum[:-1]])
        
        return seg
    
    total_len = sum(lens)
    
    all_segments = np.zeros((total_len, feature_dim), dtype=np.float32)

    ptr = 0
    for b in linkids:
        seg = get_infos(b)
        L = len(seg)
        
        all_segments[ptr:ptr+L] = seg
        ptr += L
        
    feature_dim = all_segments.shape[1]
    
    padded_clean = np.zeros((len(data), max_seq_len, feature_dim), dtype=np.float32)
    padded_aug   = np.zeros_like(padded_clean)
    
    curr_idx = 0
    Ts = []
    
    for i, l in enumerate(lens):
        seg_raw = all_segments[curr_idx : curr_idx + l].copy()
        
        seg_aug_raw, T = augment_segments(seg_raw)
        
        seg_clean_scaled = seg_raw.copy()
        seg_clean_scaled[:, 2:4] = scaler.transform(seg_clean_scaled[:, 2:4])
        padded_clean[i, :l] = seg_clean_scaled
        
        seg_aug_scaled = seg_aug_raw.copy()
        seg_aug_scaled[:, 2:4] = scaler.transform(seg_aug_scaled[:, 2:4])
        padded_aug[i, :T] = seg_aug_scaled
        
        Ts.append(T)  
        curr_idx += l
        
    augment_lens = np.array(Ts)
    
    max_aug_len = padded_aug.shape[1]
    augment_padding_mask = np.arange(max_aug_len)[None, :] >= augment_lens[:, None] # True where padding, False where data
    
    return {
        'links_clean': torch.from_numpy(padded_clean),
        'links_aug': torch.from_numpy(padded_aug),
        'augment_mask': torch.from_numpy(augment_padding_mask),
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
    new_edgeinfo = preprocess_edgeinfo(edgeinfo, args)
    
    
    # with open(os.path.join(args.absPath,args.data_config['nodes_dir']), 'rb') as f:
    #     nodeinfo = pickle.load(f)
    
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

    info_all = [new_edgeinfo, scaler]


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
    args.model_config['r_seconds'] = args.data_config["r_seconds"]
    
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

