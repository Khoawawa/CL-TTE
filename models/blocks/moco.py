# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/moco

# Modified by: yanchuan

import torch
import torch.nn as nn
import numpy as np

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, encoder_q, encoder_k, nemb, nout,
                queue_size, mmt = 0.999, temperature = 0.07, tau_I = 5):
        super(MoCo, self).__init__()

        self.queue_size = queue_size
        self.mmt = mmt
        self.temperature = temperature
        self.tau_I = tau_I

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        self.mlp_q = Projector(nemb, nout)
        self.mlp_k = Projector(nemb, nout)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(nout, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim = 0)
        

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_y", torch.zeros(queue_size))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)
        
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, y = None):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.queue_size % batch_size == 0  # for simplicity
        
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            if y is not None:
                self.queue_y[ptr:ptr + batch_size] = y
        else:
            right = self.queue_size - ptr
            self.queue[:, ptr:] = keys.T[:, :right]
            self.queue[:, :batch_size - right] = keys.T[:, right:]
            if y is not None:
                self.queue_y[ptr:] = y[:right]
                self.queue_y[:batch_size - right] = y[right:]

        # replace the keys at ptr (dequeue and enqueue)
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def masked_mean_pool(self, x, padding_mask=None):
        """
        x : (B, T, D)
        padding_mask : (B, T)  True = padded

        returns
        pooled : (B, D)
        """
        if padding_mask is None:
            return x.mean(dim=1)

        valid_mask = ~padding_mask
        valid_mask = valid_mask.unsqueeze(-1).float()

        summed = (x * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp(min=1e-6)

        return summed / counts
    def forward(self, kwargs_q, kwargs_k, y_q=None):
        mask_q = kwargs_q.get("src_key_padding_mask")
        mask_k = kwargs_k.get("src_key_padding_mask")
        # compute query features
        h = self.encoder_q(**kwargs_q)  # queries: BxTxd_model
        if not self.training:
            return None, None, h
        pooled_h = self.masked_mean_pool(h, mask_q)  # (B, d_model)
        q = self.mlp_q(pooled_h)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.mlp_k(self.masked_mean_pool(self.encoder_k(**kwargs_k), mask_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # # dequeue and enqueue
        # self._dequeue_and_enqueue(k)
        

        # return logits, labels, h

        # soft assignment
        soft_weights = None
        if y_q is not None:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                time_diff = torch.abs(y_q.float().unsqueeze(1) - self.queue_y.float().unsqueeze(0))
                soft_weights = 2 * torch.sigmoid(-self.tau_I * time_diff)
        
        self._dequeue_and_enqueue(k,y_q)
        
        # print(f"tau_I: {self.tau_I}")
        # print(f"temperature: {self.temperature}")  
        # print(f"l_pos mean: {l_pos.mean():.4f}")
        # print(f"l_neg mean: {l_neg.mean():.4f}")
        # print(f"soft_weights mean: {soft_weights.mean():.4f} min: {soft_weights.min():.4f} max: {soft_weights.max():.4f}")
        # print(f"queue_y zeros: {(self.queue_y == 0).float().mean():.3f}")

        return logits, soft_weights, h
        
    def loss(self, logits, soft_weights=None,epoch=0, max_epoch=10):
        # logits = logits / self.temperature  # make sure this is here
        log_probs = nn.functional.log_softmax(logits, dim=1)  # (N, 1+K)
        
        l_pos = -log_probs[:, 0]
        if soft_weights is None:
            return l_pos.mean()
        
        # alpha = min(1,(np.exp(epoch/max_epoch) - 1) / (np.e - 1))
        # normalize to keep scale bounded regardless of queue size
        # l_neg = -(soft_weights * log_probs[:, 1:]).sum(dim=1)
        nonzero = (soft_weights > 1e-3).float().sum(dim=1).clamp(min=1)
        l_neg = -(soft_weights * log_probs[:, 1:]).sum(dim=1) / nonzero
        print(f"l_pos: {l_pos.mean():.4f}")
        print(f"l_neg: {l_neg.mean():.4f}")
        print(f"soft_weights nonzero: {(soft_weights > 1e-3).float().sum(dim=1).mean():.1f}")
        
        return (l_pos +  l_neg).mean()


class Projector(nn.Module):
    def __init__(self, nin, nout):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(nin, nin), 
                                        nn.ReLU(), 
                                        nn.Linear(nin, nout))
        self.reset_parameter()

    def forward(self, x):
        return self.mlp(x)

    def reset_parameter(self):
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1.414)
                torch.nn.init.zeros_(m.bias)
        
        self.mlp.apply(_weights_init)
        

