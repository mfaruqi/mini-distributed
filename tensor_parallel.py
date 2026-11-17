
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import process_group_manager as pgm


# ============================================================
# PART 1: Tensor Splitting
# ============================================================

def split_tensor_along_last_dim(tensor, num_partitions):
    """Split the last dimension into equal-sized chunks."""

    # TODO 1: Find the last dimension
    last_dim = tensor.dim() - 1

    # TODO 3: Compute size of each partition
    last_dim_size = tensor.size()[last_dim] // num_partitions

    # TODO 4: Return the tensor chunks
    return torch.split(tensor, last_dim_size, dim=last_dim)


# ============================================================
# PART 2: Distributed Autograd Functions
# ============================================================

class Reduce(torch.autograd.Function):
    """All-reduce in forward, identity in backward."""

    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input

        # TODO 5: All-reduce SUM across the TP group
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.processor_group_manager.tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # TODO 6: Gradient of a sum w.r.t. each contribution
        return grad_output


class Gather(torch.autograd.Function):
    """All-gather in forward, split in backward."""

    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input

        last_dim = input.dim() - 1
        input = input.contiguous()

        tensor_list = [
            torch.empty_like(input)
            for _ in range(pgm.process_group_manager.tp_world_size)
        ]

        # TODO 7: Gather all local output shards
        # make sure input is a contigious array in memory
        input = input.contigous()
        tensor_list = [torch.empty_like(input) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = input
        dist.all_gather(tensor_list, input, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output

        # TODO 9: Divide the gradient into TP shards
        chunks = ???

        # TODO 10: Return only the current rank's shard
        return ???


class Copy(torch.autograd.Function):
    """Identity in forward, all-reduce in backward."""

    @staticmethod
    def forward(ctx, input):
        # TODO 11: Forward should leave input unchanged
        return ???

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output

        # TODO 12: Sum input-gradient contributions across TP ranks
        ???

        return grad_output


# ============================================================
# PART 3: Column Parallel Linear
# ============================================================

class ColumnParallelLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        gather_output: bool = False
    ):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features

        # TODO 13: Ensure output features divide evenly across ranks
        assert ???

        # TODO 14: Number of output features per rank
        self.output_size_per_partition = ???

        self.gather_output = gather_output

        # PyTorch weight layout: [out_features, in_features]
        # TODO 15: Create this rank's weight Parameter
        self.weight = nn.Parameter(???)

        if bias:
            # TODO 16: Create a local bias Parameter
            self.bias = nn.Parameter(???)
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.tp_world_size == 1:
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return

        # Initialize a consistent full master weight on each rank
        master_weight = torch.empty(
            self.out_features,
            self.in_features,
            dtype=self.weight.dtype,
            requires_grad=False
        )

        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # TODO 17: Split master weight along OUTPUT dimension
        weight_list = ???

        # TODO 18: Keep this TP rank's weight shard
        self.weight.data = ???

    def forward(self, input):
        # input = full activation tensor, NOT the weight matrix

        # TODO 19: Identity forward, gradient-reducing backward
        input_parallel = ???

        # TODO 20: Local linear transformation
        output = ???

        if self.gather_output:
            # TODO 21: Gather feature shards when full output is needed
            output = ???

        return output


# ============================================================
# PART 4: Row Parallel Linear
# ============================================================

class RowParallelLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool
    ):
        super().__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features

        # TODO 22: Ensure input features divide evenly
        assert ???

        # TODO 23: Number of input features per rank
        self.input_size_per_partition = ???

        # TODO 24: Create local weight Parameter
        self.weight = nn.Parameter(???)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.tp_world_size == 1:
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return

        master_weight = torch.empty(
            self.out_features,
            self.in_features,
            dtype=self.weight.dtype,
            requires_grad=False
        )

        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        # TODO 25: Split master weight along INPUT dimension
        weight_list = ???

        # TODO 26: Select this TP rank's shard
        self.weight.data = ???

    def forward(self, input):
        # input = local activation-feature shard

        # TODO 27: Compute local partial output
        output_parallel = ???

        # TODO 28: Sum partial outputs across TP ranks
        output = ???

        # TODO 29: Add bias once, after the reduce
        return ???
