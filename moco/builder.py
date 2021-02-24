# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
import paddle
import paddle.nn as nn
# import torch
# import torch.nn as nn
from .init import init_backbone_weight

class MoCo(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[0]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        init_backbone_weight(self.encoder_q)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.set_value(param_q)  # initialize
            param_k.stop_gradient = True  # not update by gradient
            # param_k.data.copy_(param_q.data)  # initialize
            # param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", paddle.randn([dim, K]))
        self.queue = nn.functional.normalize(self.queue, axis=0)

        self.register_buffer("queue_ptr", paddle.zeros([1], 'int64'))

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            # param_k = param_k * self.m + param_q * (1. - self.m)
            paddle.assign((param_k * self.m + param_q * (1. - self.m)), param_k)
            # param_k.set_value(param_k * self.m + param_q * (1. - self.m))
            param_k.stop_gradient = True
            # param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1,0])
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @paddle.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = paddle.randperm(batch_size_all).cuda()
        # idx_shuffle = pickle.load(open('/workspace/codes-vs/moco_paddle/idx_shuffle.pkl', 'rb'))
        # idx_shuffle = paddle.to_tensor(idx_shuffle)

        # broadcast to all gpus
        if paddle.distributed.get_world_size() > 1:
            # print('forward worlder size', paddle.distributed.get_world_size())
            paddle.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = paddle.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_shuffle.reshape([num_gpus, -1])[gpu_idx]
        # return paddle.gather(x_gather, idx_this), idx_unshuffle
        return paddle.index_select(x_gather, idx_this), idx_unshuffle
        # return x_gather[idx_this], idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_unshuffle.reshape([num_gpus, -1])[gpu_idx]

        return paddle.index_select(x_gather, idx_this)
        # return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, axis=1)

        # compute key features
        with paddle.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, axis=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos = paddle.sum(q * k, axis=1).unsqueeze(-1)
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = paddle.matmul(q, self.queue.clone().detach())
        # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = paddle.concat([l_pos, l_neg], axis=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = paddle.zeros([logits.shape[0]], dtype='int64')

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@paddle.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if paddle.distributed.get_world_size() < 2:
        return tensor

    # tensors_gather = [paddle.ones_like(tensor)
    #     for _ in range(paddle.distributed.get_world_size())]
    tensors_gather = []
    paddle.distributed.all_gather(tensors_gather, tensor)

    output = paddle.concat(tensors_gather, axis=0)
    return output


# Epoch: [0][ 4390/40036] Time  0.220 ( 0.288)    Data  0.000 ( 0.065)    Loss 1.0946e+01 (1.0424e+01)    Acc@1   0.00 (  0.05)  Acc@5   0.00 (  0.12)
# Epoch: [0][ 4400/40036] Time  0.200 ( 0.288)    Data  0.000 ( 0.065)    Loss 1.0931e+01 (1.0425e+01)    Acc@1   0.00 (  0.05)  Acc@5   0.00 (  0.12)
# Epoch: [0][ 4410/40036] Time  0.182 ( 0.288)    Data  0.000 ( 0.065)    Loss 1.0749e+01 (1.0426e+01)    Acc@1   0.00 (  0.05)  Acc@5   0.00 (  0.11)

# Epoch: [0][ 4390/40036] Time  0.201 ( 0.321)    Data  0.000 ( 0.097)    Loss 1.0965e+01 (1.0470e+01)    Acc@1   0.00 (  0.04)  Acc@5   0.00 (  0.11)
# Epoch: [0][ 4400/40036] Time  0.216 ( 0.321)    Data  0.000 ( 0.096)    Loss 1.0918e+01 (1.0471e+01)    Acc@1   0.00 (  0.04)  Acc@5   0.00 (  0.11)
# Epoch: [0][ 4410/40036] Time  3.413 ( 0.322)    Data  2.983 ( 0.097)    Loss 1.0877e+01 (1.0472e+01)    Acc@1   0.00 (  0.04)  Acc@5   0.00 (  0.11)


# Epoch: [0][ 4390/40036] Time  0.185 ( 0.298)    Data  0.000 ( 0.050)    Loss 1.0874e+01 (1.0458e+01)    Acc@1   0.00 (  0.04)  Acc@5   0.00 (  0.11)
# Epoch: [0][ 4400/40036] Time  0.243 ( 0.298)    Data  0.000 ( 0.050)    Loss 1.0836e+01 (1.0459e+01)    Acc@1   0.00 (  0.04)  Acc@5   0.00 (  0.11)
# Epoch: [0][ 4410/40036] Time  0.257 ( 0.298)    Data  0.000 ( 0.050)    Loss 1.0851e+01 (1.0460e+01)    Acc@1   0.00 (  0.04)  Acc@5   0.00 (  0.11)