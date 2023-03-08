from abc import abstractmethod
import torch
import torch.optim.lr_scheduler
import torch.nn as nn
import kppr.models.blocks as blocks
import kppr.models.loss as pnloss
from pytorch_lightning.core.lightning import LightningModule
from copy import deepcopy

import torch.nn.functional as F


def getModel(model_name: str, config: dict, weights: str = None):
    """Returns the model with the specific model_name. 

    Args:
        model_name ([str]): Name of the architecture (should be a LightningModule)
        config ([dict]): Parameters of the model
        weights ([str], optional): [description]. if specified: loads the weights

    Returns:
        [type]: [description]
    """
    if weights is None:
        return eval(model_name)(config)
    else:
        print(weights)
        return eval(model_name).load_from_checkpoint(weights, hparams=config)

##################################
# Base Class
##################################


class KPPR(LightningModule):
    def __init__(self, hparams: dict, data_module=None):
        super().__init__()
        hparams['batch_size'] = hparams['data_config']['batch_size']
        self.save_hyperparameters(hparams)

        self.pnloss = pnloss.EntropyContrastiveLoss(
            **hparams['loss']['params'])
        # Networks
        self.q_model = KPPRNet(hparams)
        self.k_model = deepcopy(self.q_model)
        self.k_model.requires_grad_(False)
        self.alpha = 0.999

        self.feature_bank = FeatureBank(
            size=hparams['feature_bank'], f_dim=256)
        self.feature_bank_val = FeatureBank(
            size=5000, f_dim=256)

        self.data_module = data_module

        self.top_k = [1, 5, 10]

    def forward(self, x, m):
        return self.q_model(x, m)

    def getLoss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, is_neg=None):
        return self.pnloss(anchor, positive, negative, is_neg=is_neg)

    def training_step(self, batch: dict, batch_idx):
        query = self.forward(batch['query'], batch['query_mask'])
        with torch.no_grad():
            for param_q, param_k in zip(self.q_model.parameters(), self.k_model.parameters()):
                param_k.data = param_k.data * self.alpha + \
                    param_q.data * (1. - self.alpha)

            positives = self.k_model(
                batch['positives'], batch['positives_mask'])
            negatives, is_negative = self.feature_bank.getFeatures(
                batch['neg_idx'])

        loss, losses = self.getLoss(
            query, positives, negatives, is_neg=is_negative)
        self.feature_bank.addFeatures(positives, batch['pos_idx'])

        for k, v in losses.items():
            self.log(f'train/{k}', v)

        self.log('train_loss', loss)
        recall_dict = pnloss.feature_bank_recall_nn(
            query, positives, negatives, self.top_k, is_negative)
        for k, v in recall_dict.items():
            self.log(f'train/recall_{k}', v)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        query = self.forward(batch['query'], batch['query_mask'])
        positives = self.forward(batch['positives'], batch['positives_mask'])
        negatives, is_negative = self.feature_bank_val.getFeatures(
            batch['neg_idx'])

        loss, losses = self.getLoss(
            query, positives, negatives, is_neg=is_negative)
        self.feature_bank_val.addFeatures(positives, batch['pos_idx'])

        self.log('val_loss', loss)

        for k, v in losses.items():
            self.log(f'val/{k}', v)

        recall_dict = pnloss.feature_bank_recall_nn(
            query, positives, negatives, self.top_k, is_negative)
        for k, v in recall_dict.items():
            self.log(f'val/recall_{k}', v)
        return loss

    def test_step(self, batch: dict, batch_idx):
        assert False, "test with provided test script!"

    def configure_optimizers(self):
        lr = self.hparams['train']['lr']

        optimizer = torch.optim.AdamW(
            self.q_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams['train']['max_epoch'], eta_min=self.hparams['train']['lr']/1e3)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.data_module.train_dataloader(batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return self.data_module.val_dataloader(batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return self.data_module.test_dataloader(batch_size=self.hparams['batch_size'])


#######################################################################################
######################### Perceiver ###################################################
#######################################################################################

class KPPRNet(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        # PointNet
        self.pn = blocks.PointNetFeat(
            **hparams['point_net'])

        # ConvNet
        self.conv = ConvNet(**hparams['kpconv'])

        am = hparams['aggregation']['method']
        params = hparams['aggregation'][am]
        self.aggr = blocks.VladNet(**params)

    def forward(self, x, m):
        coords = x[..., :3].clone()
        m = m.unsqueeze(-1)
        x = self.pn(x)

        x = self.conv(coords, x, m)

        x = self.aggr(x, mask=m)
        x = F.normalize(x, dim=-1)
        return x




class FeatureBank(nn.Module):
    def __init__(self, size, f_dim) -> None:
        super().__init__()
        self.register_buffer('fb', torch.full([size, f_dim], 1e8))
        self.fb = nn.functional.normalize(self.fb, dim=0)

        self.register_buffer('idx', torch.full([size], -1, dtype=torch.long))
        self.size = size

    @torch.no_grad()
    def addFeatures(self, f, idx):
        f = f.view(-1, f.shape[-1])
        idx = idx.view(-1)
        N = f.shape[0]

        self.fb = torch.roll(self.fb, N, dims=0)
        self.fb[:N] = f.detach()

        self.idx = torch.roll(self.idx, N)
        self.idx[:N] = idx

        pass

    @torch.no_grad()
    def getFeatures(self, idx):
        t = self.idx < 0
        dx = idx[..., self.idx]
        dx[..., t] = False
        return self.fb, dx


class ConvNet(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 radius,
                 num_layer=3,
                 num_neighbors=32,
                 kernel_size=3,
                 KP_extent=None,
                 p_dim=3,
                 f_dscale=2,
                 precompute_weights=True):
        super().__init__()
        in_c = [in_channels] + num_layer*[out_channels]
        self.blocks = nn.ModuleList([blocks.ResnetKPConv(
            in_channels=in_channels,
            out_channels=out_channels,
            radius=radius,
            kernel_size=kernel_size,
            KP_extent=KP_extent,
            p_dim=p_dim,
            f_dscale=f_dscale) for in_channels in in_c[:num_layer]])
        self.num_neighbors = num_neighbors
        self.num_layer = num_layer
        self.precompute_weights = precompute_weights

    def forward(self, coords: torch.Tensor, features: torch.Tensor, mask: torch.Tensor = None):
        if self.num_layer > 0:
            coords = coords.contiguous()
            coords[mask.expand_as(coords)] = 1e6
            idx = blocks.knn(coords, coords, self.num_neighbors)

            if self.precompute_weights:
                neighb_weights = self.blocks[0].precompute_weights(
                    coords, coords, idx)

            for block in self.blocks:
                if self.precompute_weights:
                    features = block.fast_forward(
                        neighb_weights, idx, features)
                else:
                    features = block(coords, coords, idx, features)
        return features
