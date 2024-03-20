##
# Thanks to the DeepCCA repo (https://github.com/Michaelvll/DeepCCA) by Zhanghao Wu (https://github.com/Michaelvll)

import torch


class DCCLoss():  #就是DCCAE 编码器对应的 loss值算法
    def __init__(self, outdim_size, device, use_all_singular_values=False):
        self.outdim_size = outdim_size #规范相关性的输出维度大小，即我们希望学习的相关性的数量。
        self.use_all_singular_values = use_all_singular_values   #一个布尔值，表示是否使用所有奇异值来计算相关性。
        self.device = device
        # print(device)

    def loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """
       #正则化的小常数，目的是增加数值计算的稳定性。
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9
       #将输入的表示 H1 和 H2 进行转置操作，以确保行表示样本。
        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        #o1 和 o2 分别表示  ，这里都设置为 H1 的行数
        o1 = o2 = H1.size(0)
        #m        这里都设置为 H1 的列数
        m = H1.size(1)
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        #[D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        D1 = torch.linalg.eigvalsh(SigmaHat11, UPLO='U')
        V1 = torch.linalg.eigh(SigmaHat11, UPLO='U').eigenvectors

        #[D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        D2 = torch.linalg.eigvalsh(SigmaHat22, UPLO='U')
        V2 = torch.linalg.eigh(SigmaHat22, UPLO='U').eigenvectors

        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            # regularization for more stability
            trace_TT = torch.add(trace_TT, (torch.eye(
                trace_TT.shape[0])*r1).to(self.device))
            #U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.linalg.eigvalsh(trace_TT, UPLO='U')
            V = torch.linalg.eigh(trace_TT, UPLO='U').eigenvectors
            U = torch.where(U > eps, U, (torch.ones(
                U.shape).double()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr
