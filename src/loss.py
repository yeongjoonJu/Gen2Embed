from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F
import itertools

class SimpleContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean') -> Tensor:
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
            
        logits = torch.matmul(x, y.transpose(0, 1))
        
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature
        self.n_target = n_target

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        
        if self.n_target > 0:
            per_query = self.n_target + 1
            target = torch.arange(
                0, dist_y.size(0), per_query, device=dist_y.device,
                dtype=torch.long
            )
            if dist_x.shape[0] != target.size(0):
                _target = torch.zeros((dist_x.size(0)), device=dist_y.device,
                   dtype=torch.long) -100
                _target[:target.size(0)] = target
                target = _target
            loss = super().__call__(dist_x, dist_y, target=target, **kwargs)
        else:
            loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.world_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)
    
    
class VarNegDistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature
        
    def gather_tensor(self, t):
        # gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        # dist.all_gather(gathered, t)
        # gathered[self.rank] = t
        # return torch.cat(gathered, dim=0)
        return self.gather_tensor_varlen(t, dim=0)
    
    def gather_tensor_varlen(self, t: torch.Tensor, dim: int = 0):
        rank       = dist.get_rank()
        world_size = dist.get_world_size()

        # ── ① 각 rank 길이 교환 ─────────────────────────
        local_len = torch.tensor([t.size(dim)], device=t.device, dtype=torch.long)
        lens = [torch.zeros_like(local_len) for _ in range(world_size)]
        dist.all_gather(lens, local_len)          # len(rank_i) → lens[i]
        lens = [l.item() for l in lens]
        max_len = max(lens)

        # ── ② zero-pad to max_len ─────────────────────
        if local_len < max_len:
            pad_shape       = list(t.shape); pad_shape[dim] = max_len - local_len
            pad_tensor      = torch.zeros(*pad_shape, device=t.device, dtype=t.dtype)
            t_padded = torch.cat([t, pad_tensor], dim=dim)
        else:
            t_padded = t                                    # 이미 max_len

        # ── ③ all_gather (autograd는 여전히 끊겨 있음) ──
        gathered = [torch.empty_like(t_padded) for _ in range(world_size)]
        dist.all_gather(gathered, t_padded)

        # ── ④ 내 rank 버퍼를 *직접* 참조로 교체해 그래프 복구 ──
        gathered[rank] = t_padded

        # ── ⑤ 패딩 제거 후 concat ───────────────────────
        out = [g.narrow(dim, 0, ln) for g, ln in zip(gathered, lens)]
        return torch.cat(out, dim=dim)


    def __call__(
        self,
        x: torch.Tensor,              # [B_local, D]
        y: torch.Tensor,              # [sum(counts)_local, D]
        batch_counts: torch.tensor,      # len=B_local, each = (1 + #negatives)
    ):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        
        # 2) all-gather the batch_counts
        all_counts = [None] * self.world_size
        dist.all_gather_object(all_counts, batch_counts)
        global_counts = list(itertools.chain.from_iterable(all_counts))
               
        # 3) global_counts 로부터 positive 인덱스(offset) 계산
        total_predicted = sum(global_counts)
        
        real_total = dist_y.size(0)
        assert total_predicted == real_total, f"{total_predicted}!={real_total}"
        
        offsets = []
        cum = 0
        for c in global_counts:
            offsets.append(cum)
            cum += c
        # offsets[i] 는 all_y 내에서 i번째 쿼리의 positive 위치
        target = torch.tensor(offsets, dtype=torch.long, device=x.device)
        
        loss = super().__call__(dist_x, dist_y, target=target)
        if self.scale_loss:
            loss = loss * self.world_size
        return loss
