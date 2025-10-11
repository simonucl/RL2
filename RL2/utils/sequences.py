from typing import List
import math
import torch
import torch.nn.functional as F
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

# TODO: write this
def gather_data(worker, minibatches):
    
    minibatches = [
        {
            k: v.to("cpu")
            for k, v in minibatch.items()
        }
        for minibatch in minibatches
    ]
    minibatches = gather_and_concat_list(
        minibatches, worker.device_mesh["dp"].get_group()
    )
    if dist.get_rank() == 0:

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