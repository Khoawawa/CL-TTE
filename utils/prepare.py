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

highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4, 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8, "primary":9, "residential":10, "primary_link":11, "unclassified":12, "tertiary":13, "secondary_link":14}
node_type = {'turning_circle':1, 'traffic_signals':2, 'crossing':3, 'motorway_junction':4, "mini_roundabout":5}

def collate_func(data, args, info_all):
    edgeinfo, nodeinfo, scaler, scaler2 = info_all

    time = torch.Tensor([d[-1] for d in data])
    linkids = []
    dateinfo = []
    inds = []
    for _, l in enumerate(data):
        linkids.append(np.asarray(l[1]))
        # dateinfo: week, date, time
        
        wday = int(l[2])
        doy = float(l[3])
        minute = float(l[4])
        doy_norm = doy / 365.0 * 2 * np.pi
        minute_norm = minute / 1440.0 * 2 * np.pi
        dateinfo.append([wday, doy_norm, minute_norm])
        inds.append(l[0])
    lens = np.asarray([len(k) for k in linkids], dtype=np.int16)
    
    def info(xs):
        infos = []
        length = 0
        for x in xs:
            info = edgeinfo[x]
            infot = []
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            infot.append(info[1])
            infot.append(length)
            length += info[1]
            try:
                infot += [nodeinfo[info[2]][0],nodeinfo[info[2]][1],nodeinfo[info[3]][0],nodeinfo[info[3]][1]]
            except:
                print(info)
            infos.append(np.asarray(infot))
            # highway length sumoflength gps4

        return infos

    con_links = np.concatenate([info(b) for b in linkids], dtype='object')
    # print(merge_start_mask.shape, merge_pad_mask.shape)
    mask = np.arange(lens.max()) < lens[:, None]
    padded = np.zeros((*mask.shape, 1+2+4), dtype=np.float32)
    con_links[:, 1:3] = scaler.transform(con_links[:, 1:3])
    con_links[:, 3:7] = scaler2.transform(con_links[:, 3:7])

    padded[mask] = con_links
    
    return {'links':torch.from_numpy(padded),
            'dateinfo': torch.from_numpy(np.asarray(dateinfo, dtype=np.float32)),
            'lens':torch.LongTensor(lens), 
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
    else:
        ValueError("Wrong Dataset Name")

    info_all = [edgeinfo, nodeinfo, scaler, scaler2]


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
    return Cl_TTE(**model_config)
        

def create_main_loss(loss_cl,loss_eta, args):
    beta = args.beta
    
    # scale = (loss_eta.detach() / (loss_cl.detach() + 1e-6))
    scale = 1 / (loss_cl / loss_eta + 1e-4).detach()
    # loss_cl_scaled = loss_cl * scale.clamp(0.1, 10.0)   
    loss_cl_scaled = loss_cl * scale.clamp(0.1, 10.0)
    
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

