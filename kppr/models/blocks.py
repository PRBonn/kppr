import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter
import numpy as np
import math
import opt_einsum as oe


class VladNet(nn.Module):
    def __init__(self, feature_dim, nr_center=64, out_dim=256, norm=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.nr_center = nr_center
        self.softmax = nn.Softmax(dim=-1)
        self.sm_center = nn.Linear(feature_dim, nr_center)
        self.center = nn.Parameter(torch.randn(
            feature_dim, nr_center) / feature_dim**0.5)
        self.feature_proj = nn.Linear(
            feature_dim*nr_center, out_dim, bias=False)

    def forward(self, x: torch.Tensor, mask=None):
        """Computes the Vlad [..., out] with learned clusters for an input x [...,n, fd]

        Args:
            x (torch.Tensor): Features [...,n,fd]
        """
        a = self.sm_center(
            x)  # quadratic distance from each point to each center
        a = self.softmax(a)  # softmax to get the weights
        if mask is not None:
            a = a*torch.logical_not(mask)
        a_sum = a.sum(dim=-2, keepdim=True)
        center_weighted = self.center * a_sum  # reweight the centers

        a = a.transpose(-2, -1)
        x_weighted = torch.matmul(a, x).transpose(-2, -1)
        vlad = (x_weighted - center_weighted)
        vlad = F.normalize(vlad, dim=-1, p=2)
        shape = x.shape[:-2]+(self.nr_center * self.feature_dim,)
        vlad = vlad.reshape(shape)
        vlad = F.normalize(vlad, dim=-1, p=2)
        vlad = self.feature_proj(vlad)
        return vlad


class STNkd(nn.Module):
    def __init__(self, k=64, norm=True):
        super(STNkd, self).__init__()
        self.conv1 = nn.Linear(k, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        # exchanged Batchnorm1d by Layernorm
        self.bn1 = nn.LayerNorm(64) if norm else nn.Identity()
        self.bn2 = nn.LayerNorm(128) if norm else nn.Identity()
        self.bn3 = nn.LayerNorm(1024) if norm else nn.Identity()
        self.bn4 = nn.LayerNorm(512) if norm else nn.Identity()
        self.bn5 = nn.LayerNorm(256) if norm else nn.Identity()

        self.k = k

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, -2, keepdim=True)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device, dtype=x.dtype)
        shape = x.shape[:-1]+(1,)
        iden = iden.repeat(*shape)
        x = x.view(iden.shape) + iden
        return x


class PointNetFeat(nn.Module):
    def __init__(self, in_dim=3, out_dim=1024, feature_transform=False, norm=True):
        super(PointNetFeat, self).__init__()
        self.stn = STNkd(k=in_dim, norm=norm)
        self.conv1 = nn.Linear(in_dim, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, out_dim)
        self.bn1 = nn.LayerNorm(64) if norm else nn.Identity()
        self.bn2 = nn.LayerNorm(128) if norm else nn.Identity()
        self.bn3 = nn.LayerNorm(out_dim) if norm else nn.Identity()
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64, norm=norm)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.matmul(x, trans_feat)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x

############################
###### KPConv stuff ########
############################
# mostly taken from Thomas Hugues repo https://github.com/HuguesTHOMAS/KPConv-PyTorch

def knn(q_pts, s_pts, k, cosine_sim=False):
    if cosine_sim:
        sim = torch.einsum('...in,...jn->...ij', q_pts, s_pts)
        _, neighb_inds = torch.topk(sim, k, dim=-1, largest=True)
        return neighb_inds

    else:
        dist = ((q_pts.unsqueeze(-2) - s_pts.unsqueeze(-3))**2).sum(-1)
        _, neighb_inds = torch.topk(dist, k, dim=-1, largest=False)
        return neighb_inds


def vector_gather(vectors: torch.Tensor, indices: torch.Tensor):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[B, N1, D]
        indices: Tensor[B, N2, K]
    Returns:
        Tensor[B,N2, K, D]
    """

    # src
    vectors = vectors.unsqueeze(-2)
    shape = list(vectors.shape)
    shape[-2] = indices.shape[-1]
    vectors = vectors.expand(shape)

    # Do the magic
    shape = list(indices.shape)+[vectors.shape[-1]]
    indices = indices.unsqueeze(-1).expand(shape)
    out = torch.gather(vectors, dim=-3, index=indices)
    return out


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


class KPConv(nn.Module):

    def __init__(self, in_channels, out_channels, radius, kernel_size=3, KP_extent=None, p_dim=3):
        """
        Initialize parameters for KPConvDeformable.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param radius: radius used for kernel point init.
        :param kernel_size: Number of kernel points.
        :param KP_extent: influence radius of each kernel point. (float), default: None
        :param p_dim: dimension of the point space. Default: 3
        :param radial: bool if direction independend convolution 
        :param align_kp: aligns the kernel points along the main directions of the local neighborhood 
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.p_dim = p_dim  # 1D for radial convolution

        self.K = kernel_size ** self.p_dim
        self.num_kernels = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = radius / (kernel_size-1) * \
            self.p_dim**0.5 if KP_extent is None else KP_extent

        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a grid
        :return: the tensor of kernel points
        """

        K_points_numpy = self.getKernelPoints(self.radius,
                                              self.num_kernels, dim=self.p_dim)

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)


    def getKernelPoints(self, radius, num_points=3, dim=3):
        """[summary]

        Args:
            radius (float): radius
            num_points (int, optional): Number of kernel points per dimension. Defaults to 3.

        Returns:
            [type]: returns num_points^3 kernel points 
        """
        xyz = np.linspace(-1, 1, num_points)
        if dim == 1:
            return xyz[:, None]*radius

        points = np.meshgrid(*(dim*[xyz]))
        points = [p.flatten() for p in points]
        points = np.vstack(points).T
        points /= dim**(0.5)  # Normalizes to stay in unit sphere
        return points*radius

    def precompute_weights(self, q_pts, s_pts, neighb_inds):
        s_pts = torch.cat(
            (s_pts, torch.zeros_like(s_pts[..., :1, :]) + 1e6), -2)

        # Get neighbor points and features [n_points, n_neighbors, dim/ in_fdim]
        if len(neighb_inds.shape) < 3:
            neighbors = s_pts[neighb_inds, :]
        else:
            neighbors = vector_gather(s_pts, neighb_inds)

        # Center every neighborhood [n_points, n_neighbors, dim]
        neighbors = neighbors - q_pts.unsqueeze(-2)

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        
        kernel_points = self.kernel_points
        neighbors.unsqueeze_(-2)

        differences = neighbors - kernel_points
        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=-1)
        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        all_weights = torch.clamp(
            1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
        return all_weights

    @staticmethod
    def gather_features(neighb_inds, x):
        x = torch.cat((x, torch.zeros_like(x[..., :1, :])), -2)
        if len(neighb_inds.shape) < 3:
            return gather(x, neighb_inds)
        else:
            return vector_gather(x, neighb_inds)

    def convolution(self, neighb_weights, neighb_x):
        fx = oe.contract('...nkl,...nki,...lio->...no',
                         neighb_weights, neighb_x, self.weights)
        return fx

    def forward(self, q_pts, s_pts, neighb_inds, x):
        # Add a fake point/feature in the last row for shadow neighbors
        s_pts = torch.cat(
            (s_pts, torch.zeros_like(s_pts[..., :1, :]) + 1e6), -2)

        # Get neighbor points and features [n_points, n_neighbors, dim/ in_fdim]
        x = torch.cat((x, torch.zeros_like(x[..., :1, :])), -2)
        if len(neighb_inds.shape) < 3:
            neighbors = s_pts[neighb_inds, :]
            neighb_x = gather(x, neighb_inds)
        else:
            neighbors = vector_gather(s_pts, neighb_inds)
            neighb_x = vector_gather(x, neighb_inds)

        # Center every neighborhood [n_points, n_neighbors, dim]
        neighbors = neighbors - q_pts.unsqueeze(-2)

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        
        # print('neighbors', neighbors.shape)
        kernel_points = self.kernel_points
        neighbors.unsqueeze_(-2)

        # print(kernel_points.shape,neighbors.shape)
        differences = neighbors - kernel_points
        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=-1)
        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        all_weights = torch.clamp(
            1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)

        # fx = torch.einsum('...nkl,...nki,...lio->...no',
        #                   all_weights, neighb_x, self.weights)
        fx = oe.contract('...nkl,...nki,...lio->...no',
                         all_weights, neighb_x, self.weights)
        return fx

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)


class ResnetKPConv(nn.Module):
    def __init__(self, in_channels, out_channels, radius, kernel_size=3, KP_extent=None, p_dim=3, f_dscale=2):
        super().__init__()

        self.ln1 = nn.LayerNorm(in_channels)
        self.relu = nn.LeakyReLU()

        self.kpconv = KPConv(in_channels=in_channels,
                             out_channels=out_channels//f_dscale,
                             radius=radius,
                             kernel_size=kernel_size,
                             KP_extent=KP_extent,
                             p_dim=p_dim)

        self.ln2 = nn.LayerNorm(out_channels//f_dscale)
        self.lin = nn.Linear(out_channels//f_dscale, out_channels)

        self.in_projection = nn.Identity() if in_channels == out_channels else nn.Linear(
            in_channels, out_channels)

    def forward(self, q_pts, s_pts, neighb_inds, x):
        xr = self.relu(self.ln1(x))
        xr = self.kpconv(q_pts, s_pts, neighb_inds, x)
        xr = self.relu(self.ln2(xr))
        xr = self.lin(xr)

        return self.in_projection(x) + xr

    @torch.no_grad()
    def precompute_weights(self, q_pts, s_pts, neighb_inds):
        return self.kpconv.precompute_weights(q_pts, s_pts, neighb_inds)

    def fast_forward(self, neighb_weights, neighb_inds, x):
        xr = self.relu(self.ln1(x))

        neighb_x = self.kpconv.gather_features(neighb_inds, xr)
        xr = self.kpconv.convolution(neighb_weights, neighb_x)

        xr = self.relu(self.ln2(xr))
        xr = self.lin(xr)

        return self.in_projection(x) + xr

