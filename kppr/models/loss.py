import torch.nn as nn
import torch


def repulsion_loss(sim_ap: torch.Tensor, sim_an: torch.Tensor, is_neg, max_dist=1):
    sim_an[~is_neg] = -1000
    dist_p = (1 - sim_ap.max(-1)[0])/2
    dist_n = (1 - sim_an.max(-1)[0])/2
    dist = torch.where(dist_p < dist_n, dist_p, dist_n) + 1e-5
    dist[dist > max_dist] = 1
    return -dist.log().mean()

class EntropyContrastiveLoss(nn.Module):
    def __init__(self, margin, alpha=0) -> None:
        super().__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, q, p, n, is_neg):
        sp = torch.einsum('...n,...n->...', q, p)
        sn = torch.einsum('...n,...n->...', q, n)
        l_rep = repulsion_loss(sp, sn, is_neg, max_dist=0.1) * self.alpha

        within_margin = sn > self.margin  # only compute loss if similarity is over margin

        l_c = (1-sp).mean(-1) + (sn * is_neg *
                                 within_margin).sum(-1) / (is_neg * within_margin + 1e-5).sum(-1)
        l_c = l_c.mean()
        losses = {'l_cont': l_c, 'l_rep': l_rep}
        loss = l_c + l_rep
        return loss, losses

def feature_bank_recall(q, p, n, top_k: list, is_neg=None):
    tk = [k-1 for k in top_k]
    sim_pos = torch.einsum('...n,...n->...', q, p)
    sim_pos, _ = sim_pos.max(-1)  # get the easiest positive
    sim_neg = torch.einsum('...n,...n->...', q, n) * \
        is_neg  # set similarity of not neg to 0
    top_k_neg, i = torch.topk(sim_neg, top_k[-1], dim=-1)
    top_k_neg = top_k_neg[..., tk]
    recall = sim_pos[..., None] > top_k_neg
    recall = recall.float().mean(0)
    recall_dict = {k: r for k, r in zip(top_k, recall)}
    return recall_dict


def feature_bank_recall_nn(q, p, n, top_k: list, is_neg=None):
    tk = [k-1 for k in top_k]
    sim_pos = 1/((q - p)**2).sum(-1)
    sim_pos, _ = sim_pos.max(-1)  # get the easiest positive
    sim_neg = 1/((q - n.unsqueeze(-3))**2).sum(-1) * \
        is_neg  # set similarity of not neg to 0
    top_k_neg, i = torch.topk(sim_neg, top_k[-1], dim=-1)
    top_k_neg = top_k_neg[..., tk]
    recall = sim_pos[..., None] > top_k_neg
    recall = recall.float().mean(0)
    recall_dict = {k: r for k, r in zip(top_k, recall)}
    return recall_dict
