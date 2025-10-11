import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams

def slide_tensor_along_cp(tensor):

    cp_rank = mpu.get_context_parallel_rank()
    multiple_of = 2 * mpu.get_context_parallel_world_size()
    if len(tensor) % multiple_of != 0:
        pad_tokens = multiple_of - len(tensor) % multiple_of
        tensor = F.pad(tensor, (0, pad_tokens), value=0)
    chunk_size = len(tensor) // multiple_of
    start_1, end_1 = cp_rank * chunk_size, (cp_rank + 1) * chunk_size
    start_2, end_2 = (multiple_of - cp_rank - 1) * chunk_size, (multiple_of - cp_rank) * chunk_size
    return torch.cat((tensor[start_1:end_1], tensor[start_2:end_2]))

def slide_along_cp(minibatch):

    seq_lens = minibatch["eos_mask"].argmax(-1) + 1
    multiple_of = mpu.get_tensor_model_parallel_world_size()
    processed_minibatch = {}
    for k, v in minibatch.items():
        tensors = [
            slide_tensor_along_cp(tensor[:seq_len])
            for tensor, seq_len in zip(v, seq_lens)
        ]
        length = sum([len(tensor) for tensor in tensors])
        if length % multiple_of != 0:
            pad_tokens = multiple_of - length % multiple_of
            tensors.append(
                torch.zeros(
                    (pad_tokens), dtype=v.dtype, device=v.device
                )
            )
        processed_minibatch[k] = torch.cat(tensors).unsqueeze(0)
    seq_lens = torch.LongTensor([
        len(tensor) for tensor in tensors
    ])
    cu_seqlens = torch.cumsum(
        torch.cat((torch.LongTensor([0]), seq_lens)),
        dim=0,
        dtype=torch.int32
    ).to(torch.cuda.current_device())
    cu_seqlens = mpu.get_context_parallel_world_size() * cu_seqlens
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd"
    )
    return processed_minibatch, packed_seq_params

def gather_along_cp(minibatch, packed_seq_lens):

    cp_size = mpu.get_context_parallel_world_size()
    cu_seqlens = packed_seq_lens.cu_seqlens_q // cp_size
    processed_minibatch = {}
    for k, v in minibatch.items():
        v = v.squeeze(0)
        processed_tensors = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            tensor = v[start:end]
            tensors = [
                torch.zeros_like(tensor) for _ in range(cp_size)
            ]
            dist.all_gather(
                tensors,
                tensor,
                group=mpu.get_context_parallel_group()
            )
            tensors[mpu.get_context_parallel_rank()] = tensor # necessary to retain grad
            tensor = torch.cat(
                [
                    tensor[:len(tensor) // 2] for tensor in tensors
                ] + [
                    tensor[len(tensor) // 2:] for tensor in tensors[::-1]
                ]
            )
            processed_tensors.append(tensor)
        processed_minibatch[k] = pad_sequence(processed_tensors, True)
    return processed_minibatch