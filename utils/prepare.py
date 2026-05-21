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

def augment_segments(
    seg,
    p_poi=0.05,
    p_seg=0.05,
    length_noise_std=0.01,
):
    """
    Conservative SSL augmentation.

    Assumes cumulative length is NOT used by contrastive learning.
    """

    seg_aug = seg.copy().astype(np.float32)

    T, F = seg_aug.shape

    # =========================================================
    # MILD POI DROPOUT
    # =========================================================

    poi = seg_aug[:, 4:]

    mask_poi_rows = (
        np.random.rand(T, 1) < p_poi
    )

    poi = poi * (~mask_poi_rows)

    seg_aug[:, 4:] = poi

    # =========================================================
    # VERY LIGHT FEATURE MASKING
    # =========================================================

    seg_mask = np.random.rand(T) < p_seg

    if seg_mask.all():
        seg_mask[np.random.randint(T)] = False
    seg_aug[seg_mask, 0:2] = 1 # highway -> unclassified
    # speed bucket -> unknown (len(SPEED_BUCKETS))
    seg_aug[seg_mask, 3] = len(SPEED_BUCKETS)
    # remove poi context
    seg_aug[seg_mask, 4:] = 0

    # =========================================================
    # SMALL LENGTH NOISE
    # =========================================================

    noise = np.random.normal(
        loc=1.0,
        scale=length_noise_std,
        size=seg_aug[:, 2].shape
    )

    seg_aug[:, 2] *= noise

    return seg_aug, T

def preprocess_edgeinfo(edgeinfo, args):
    """Returns both the dict (legacy) and a dense numpy matrix for fast batch lookup."""
    n_poi = args.data_config['n_poi_groups']
    feature_dim = 2 + 1 + 1 + n_poi  # hw(2), length(1), speed(1), poi(n)

    new_edgeinfo = {}
    max_id = max(edgeinfo.keys()) if edgeinfo else 0
    # +1 row as a zero-pad for edge id 0 / missing
    edge_matrix = np.zeros((max_id + 2, feature_dim), dtype=np.float32)
    edge_id_col = np.zeros(max_id + 2, dtype=np.int64)

    for k, info in edgeinfo.items():
        hw_ids     = parse_highway_tags(info[0])
        speed_buck = parse_maxspeed(info[4 + n_poi])
        poi_self   = (np.array(info[4:4 + n_poi], dtype=np.float32) > 0).astype(np.float32)

        row = np.array([*hw_ids, info[1], speed_buck, *poi_self], dtype=np.float32)
        edge_matrix[k] = row
        edge_id_col[k] = k

        new_edgeinfo[k] = [hw_ids, info[1], speed_buck, *poi_self, k]

    return new_edgeinfo, edge_matrix, edge_id_col
def precompute_overlap_mask(dataset, max_edge_id):
    """
    Builds a boolean array: overlap[i, j] = True if trips i and j share >80% edges.
    Runs once on CPU at startup. Stored as a uint8 array to save RAM.
    """
    N = len(dataset)
    # represent each trip as a set
    trip_sets = []
    for d in dataset:
        ids = set(int(x) for x in d[1] if x != 0)
        trip_sets.append(ids)

    # sparse storage — only store the True pairs
    overlap = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        si = trip_sets[i]
        li = len(si)
        if li == 0:
            continue
        for j in range(i + 1, N):
            sj = trip_sets[j]
            inter = len(si & sj)
            ratio = inter / (min(li, len(sj)) + 1e-6)
            if ratio > 0.8:
                overlap[i, j] = True
                overlap[j, i] = True

    return overlap

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
    edgeinfo, edge_matrix, edge_id_col, scaler, overlap_matrix = info_all  # <-- unpack new fields

    time      = torch.Tensor([d[-1] for d in data])
    dateinfo  = [[int(l[2]), float(l[3]), float(l[4])] for l in data]
    inds      = [l[0] for l in data]

    # ignore_mask_np = overlap_matrix[np.ix_(inds, inds)]  # (B, B) boolean numpy array
    max_allowed_len = 128
    linkids   = [np.asarray(l[1])[:max_allowed_len] for l in data]
    lens        = np.array([len(k) for k in linkids])
    max_seq_len = lens.max()
    feature_dim = edge_matrix.shape[1]

    padded_clean   = np.zeros((len(data), max_seq_len, feature_dim), dtype=np.float32)
    padded_culm    = np.zeros((len(data), max_seq_len, 1),           dtype=np.float32)
    padded_edgeids = np.zeros((len(data), max_seq_len),              dtype=np.int64)

    for i, xs in enumerate(linkids):
        L = len(xs)
        # ---- single vectorized lookup instead of a Python loop ----
        seg                = edge_matrix[xs]          # (L, F)
        padded_edgeids[i, :L] = edge_id_col[xs]

        raw_lengths = seg[:, 2].copy()                # length column, before scaling
        cum         = np.cumsum(raw_lengths)
        culm_len    = np.concatenate([[0.0], cum[:-1]])

        length_and_culm  = np.stack([seg[:, 2], culm_len], axis=-1)
        length_and_culm  = scaler.transform(length_and_culm)
        seg[:, 2]        = length_and_culm[:, 0]
        culm_len         = length_and_culm[:, 1]

        padded_clean[i, :L]    = seg
        padded_culm[i, :L, 0]  = culm_len

    # augmentation — batch it all at once instead of per-sample loop
    padded_aug = padded_clean.copy()
    for i, l in enumerate(lens):
        seg_aug, _ = augment_segments(padded_clean[i, :l])
        padded_aug[i, :l] = seg_aug

    return {
        'links_clean': torch.from_numpy(padded_clean),
        'links_aug':   torch.from_numpy(padded_aug),
        'dateinfo':    torch.from_numpy(np.asarray(dateinfo, dtype=np.float32)),
        'culm_len':    torch.from_numpy(padded_culm),
        # 'ignore_mask': torch.from_numpy(ignore_mask_np),
        'lens':        torch.LongTensor(lens),
        'inds':        inds,
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
class VarianceBucketSampler:
    def __init__(self, dataset, batch_size, pool_factor=10):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.pool_factor = pool_factor # Controls the Padding vs. Variance trade-off
        
        if isinstance(dataset[0], dict):
            self.lengths = [len(d['lats']) for d in dataset]
        elif isinstance(dataset[0][1], list):
            self.lengths = [len(d[1]) for d in dataset]
        else:
            self.lengths = [d[0]['lens'] for d in dataset]
            
        self.indices = list(range(self.count))

    def __iter__(self):
        # 1. Global shuffle first
        np.random.shuffle(self.indices)

        # 2. Divide into massive chunks (same as before)
        chunk_size = self.batch_size * 100
        chunks = (self.count + chunk_size - 1) // chunk_size

        for i in range(chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, self.count)
            chunk_idx = self.indices[start:end]
            
            # 3. Sort the mega-chunk by length (This bounds our max padding)
            chunk_idx.sort(
                key=lambda x: (
                    self.lengths[x] + np.random.randint(-10, 10)
                ),
                reverse=True
            )
            
            # 4. THE FIX: Sub-pool shuffling
            # Group into pools of e.g. 10 batches. 
            # Shuffle INSIDE the pool before yielding to guarantee variance.
            window = self.batch_size * 4

            for s in range(0, len(chunk_idx), window):

                sub = chunk_idx[s:s+window]

                np.random.shuffle(sub)

                chunk_idx[s:s+window] = sub
                
            self.indices[start:end] = chunk_idx

        # 5. Yield the finalized batches
        batches = (self.count + self.batch_size - 1) // self.batch_size
        for i in range(batches):
            yield self.indices[i * self.batch_size: min((i + 1) * self.batch_size, self.count)]

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
    new_edgeinfo, edge_matrix, edge_id_col = preprocess_edgeinfo(edgeinfo, args)
    
    tdata_train = np.load(os.path.join(args.absPath, args.data_config['data_dir'], 'train.npy'), allow_pickle=True)
    # overlap_matrix = precompute_overlap_mask(tdata_train, max_edge_id=edge_matrix.shape[0]-1)
    overlap_matrix = None
    
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

    info_all = [new_edgeinfo, edge_matrix, edge_id_col, scaler, overlap_matrix]


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
            loader[phase] = DataLoader(
                Datadict(data[phase]), 
                batch_sampler=VarianceBucketSampler(data[phase], args.batch_size, pool_factor=5),
                collate_fn=lambda x: collate_func(x, args, info_all),
                pin_memory=True,
                num_workers=2,
                persistent_workers=True
            )
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
    args.model_config['tau_I'] = args.data_config["tau_I"]
    
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

