from typing import List
import math
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from RL2.utils.commication import gather_and_concat_list
from RL2.utils.seqlen_balance import get_seqlen_balanced_partitions

def tensor_dict_to_minibatches(
    tensor_dict,
    dp_size,
    max_length_per_dp,
    pair: bool
):

    # We pack sequences into minibatches for higher throughput.
    # There are two constrains:
    #   * The length of any minibatch cannot exceed `max_length_per_dp`
    #   * The number of minibatches must be multiple of dp size (so that
    #     each dp shares identical number of minibatches)
    # To satisfy the first constraint, the number of minibatches must be
    # at least `math.ceil(total_length / max_length_per_dp)`.
    # Starting from the first multiple of dp size that is no less than 
    # the value, we pack sequences into `n_minibatches` minibatches and 
    # check whether the first constraint is satisfied. If not, we increase 
    # `n_minibatches` by dp size (so that the second constraint is always 
    # satisfied) and repeat the loop.
    seq_len_list = (tensor_dict["eos_mask"].argmax(-1) + 1).tolist()
    if pair:
        # When pair, every two adjacent sequences will be colocated, so 
        # their length are summed.
        seq_len_list = torch.tensor(seq_len_list).view(-1, 2).sum(-1).tolist()
    assert max(seq_len_list) <= max_length_per_dp, \
        f"The longest sequence has a length of {max(seq_len_list)}," \
        f"which exceeds the maximum length per dp {max_length_per_dp}."
    n_minibatches = math.ceil(sum(seq_len_list) / max_length_per_dp)
    if n_minibatches % dp_size != 0:
        n_minibatches += dp_size - n_minibatches % dp_size

    # Partition sequences into n_minibatches balanced minibatches.
    while True:

        global PAD_SEQUENCES
        if n_minibatches > len(seq_len_list):
            # The number of sequences must be no less than `n_minibatches`.
            # If not, we pad the number of sequences to `n_minibatches`.
            PAD_SEQUENCES = n_minibatches - len(seq_len_list)
            for k, v in tensor_dict.items():
                tensor_dict[k] = F.pad(
                    v,
                    (0, 0, 0, (2 if pair else 1) * PAD_SEQUENCES),
                    value=0
                )
            seq_len_list.extend(PAD_SEQUENCES * [0])
        else:
            PAD_SEQUENCES = 0

        partitions: List[List[int]] = get_seqlen_balanced_partitions(
            seq_len_list, k_partitions=n_minibatches, equal_size=False
        )
        max_minibatch_length = max([
            sum([seq_len_list[p] for p in partition])
            for partition in partitions
        ])
        if max_minibatch_length <= max_length_per_dp:
            break
        n_minibatches += dp_size

    if pair:
        partitions = [
            sum([[2 * p, 2 * p + 1] for p in partition], [])
            for partition in partitions
        ]
    global SHUFFLE_INDICES
    SHUFFLE_INDICES = sum(partitions, [])

    return [
        {
            k: v[partition] for k, v in tensor_dict.items()
        }
        for partition in partitions
    ]

def scatter_data(
    tensor_dict,
    process_group,
    max_length_per_dp,
    num_batches : int = 1,
    pair : bool = False
):

    if num_batches > 1:
        if dist.get_rank() == 0:
            batch_size = math.ceil(len(tensor_dict["states"]) / num_batches)
            batches = []
            for num_batch in range(num_batches):
                batch_tensor_dict = {
                    k: v[num_batch * batch_size:(num_batch + 1) * batch_size]
                    for k, v in tensor_dict.items()
                }
                batches.append(
                    scatter_data(
                        batch_tensor_dict, process_group, max_length_per_dp, pair=pair
                    )
                )
            return batches
        else:
            return [
                scatter_data(None, process_group, max_length_per_dp, pair=pair)
                for _ in range(num_batches)
            ]

    dp_rank = dist.get_rank(process_group)
    dp_size = dist.get_world_size(process_group)
    if dist.get_rank() == 0:
        minibatches = tensor_dict_to_minibatches(
            tensor_dict, dp_size, max_length_per_dp, pair
        )
    object_list = [minibatches] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(
        object_list,
        src=0
    )
    minibatches = object_list[0]
    chunk_size = len(minibatches) // dp_size
    minibatches = minibatches[dp_rank * chunk_size:(dp_rank + 1) * chunk_size]
    return [
        {
            k: v.to(torch.cuda.current_device())
            for k, v in minibatch.items()
        }
        for minibatch in minibatches
    ]

def gather_data(minibatches, process_group):
    
    minibatches = [
        {
            k: v.to("cpu")
            for k, v in minibatch.items()
        }
        for minibatch in minibatches
    ]
    minibatches = gather_and_concat_list(
        minibatches, process_group
    )
    if dist.get_rank() == 0:
        
        length = max([
            minibatch["eos_mask"].argmax(-1).max().item()
            for minibatch in minibatches
        ])
        minibatches = [
            {k: v[:, :length] for k, v in minibatch.items()}
            for minibatch in minibatches
        ]
        tensor_dict = {
            k: torch.cat([
                minibatch[k] for minibatch in minibatches
            ])
            for k in minibatches[0].keys()
        }

        reversed_indices = len(SHUFFLE_INDICES) * [None]
        for idx, shuffle_idx in enumerate(SHUFFLE_INDICES):
            reversed_indices[shuffle_idx] = idx
        tensor_dict = {
            k: v[reversed_indices]
            for k, v in tensor_dict.items()
        }

        if PAD_SEQUENCES > 0:
            tensor_dict = {
                k: v[:-PAD_SEQUENCES]
                for k, v in tensor_dict.items()
            }

        return tensor_dict

def count_total(minibatches, key, process_group):

    if isinstance(key, tuple):
        return tuple(
            count_total(minibatches, k, process_group)
            for k in key
        )
        
    total = sum(
        [minibatch[key].sum() for minibatch in minibatches]
    )
    total = torch.Tensor(
        [total]
    ).to(torch.cuda.current_device())
    dist.all_reduce(
        total,
        op=dist.ReduceOp.SUM,
        group=process_group
    )
    return total.to("cpu").item()

def slide_along_cp(minibatch, process_group, multiple_of):

    cp_rank = dist.get_rank(process_group)
    cp_size = dist.get_world_size(process_group)
    def slide_tensor_along_cp(tensor):

        if len(tensor) % (2 * cp_size) != 0:
            pad_tokens = 2 * cp_size - len(tensor) % (2 * cp_size)
            tensor = F.pad(tensor, (0, pad_tokens), value=0)
        chunk_size = len(tensor) // (2 * cp_size)
        start_1, end_1 = cp_rank * chunk_size, (cp_rank + 1) * chunk_size
        start_2, end_2 = (2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size
        return torch.cat((tensor[start_1:end_1], tensor[start_2:end_2]))

    seq_lens = minibatch["eos_mask"].argmax(-1) + 1
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
    return processed_minibatch, cu_seqlens

def gather_along_cp(minibatch, process_group, cu_seqlens):
    
    cp_rank = dist.get_rank(process_group)
    cp_size = dist.get_world_size(process_group)
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
                group=process_group
            )
            tensors[cp_rank] = tensor # necessary to retain grad
            inorder_tensors = [
                tensor[:len(tensor) // 2] for tensor in tensors
            ]
            reversed_tensors = [
                tensor[len(tensor) // 2:] for tensor in tensors[::-1]
            ]
            tensor = torch.cat(inorder_tensors + reversed_tensors)
            processed_tensors.append(tensor)
        processed_minibatch[k] = pad_sequence(processed_tensors, True)
    if processed_minibatch["eos_mask"][-1].max().item() == 0:
        processed_minibatch = {
            k: v[:-1] for k, v in processed_minibatch.items()
        }
    return processed_minibatch