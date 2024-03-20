import os
import torch
import torch.nn.functional as F
import multiprocessing

from torch import nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image


class LSTMEncoder(nn.Module):  #用于将输入序列转换为一个低维的表示（这里用的是单层）
    def __init__(self, input_size, representation_size, num_layers=1, batch_first=True): #定义输入参数：输入数据的大小，编码器输出维度，LSTM层数（默认为1），...
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=representation_size,
                            num_layers=num_layers, batch_first=batch_first)
        nn.init.orthogonal_(self.lstm.weight_ih_l0) #对LSTM层的输入权重和隐藏权重进行正交初始化。这有助于防止梯度爆炸或梯度消失问题
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        out, _ = self.lstm(x)  #输出（压缩的低维度向量）
        return out


class LSTMDecoder(nn.Module):  #LSTM的解码器，把从编码器得到的低维度数据转为高纬度（原来的数据维度）
    def __init__(self, representation_size, output_size, num_layers=1, batch_first=True):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=representation_size, hidden_size=output_size,
                            num_layers=num_layers, batch_first=batch_first)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class LSTMAutoEncoder(nn.Module):  #forward函数返回的是解码器预测值（与训练编码器有关）、encode函数就是输出经过编码器后的输出序列
    def __init__(self, input_size, representation_size, num_layers=1, batch_first=True):
        super(LSTMAutoEncoder, self).__init__()
        self.batch_first = batch_first
        self.encoder = LSTMEncoder(
            input_size=input_size, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder = LSTMDecoder(representation_size=representation_size,
                                   output_size=input_size, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x):
        #获取输入序列的长度。
        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        out = self.encoder(x)
        representation = out[:, -1,
                             :].unsqueeze(1) if self.batch_first else out[-1, :, :].unsqueeze(0)  #获取编码后序列的最后一个时间步的表示，作为重构的表示。
        #print(f"representation:{representation}")
        representation_seq = representation.expand(-1, seq_len, -1)  #扩展
        #print(f"representation_seq:{representation_seq}")
        x_prime = self.decoder(representation_seq)                 #得到解码器的预测输出值
        return x_prime

    def encode(self, x):        #单纯的使用编码器
        x = self.encoder(x)
        return x


class DCCLSTMAutoEncoder(nn.Module):   #只是单纯的输出了隐藏表示层和预测输出层，相关损失计算未在这里（两个通道，分别单独处理A和B两个模态，得到的是两个对应的隐藏表示输出（A是A，B是B））
    def __init__(self, input_size_A, input_size_B, representation_size, num_layers=1, batch_first=True):
        super(DCCLSTMAutoEncoder, self).__init__()
        self.batch_first = batch_first
        # 定义模态A的LSTM编码器和解码器
        self.encoder_A = LSTMEncoder(
            input_size=input_size_A, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder_A = LSTMDecoder(representation_size=representation_size,
                                     output_size=input_size_A, num_layers=num_layers, batch_first=batch_first)
        # 定义模态A的LSTM编码器和解码器
        self.encoder_B = LSTMEncoder(
            input_size=input_size_B, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder_B = LSTMDecoder(representation_size=representation_size,
                                     output_size=input_size_B, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x_A=None, x_B=None):  #保持两个模态直接的相关性
        """Takes the input from two modalities and forwards.  接收两个模态的输入并进行前向传播

        Args:
            x_A: input tensor of modality A
            x_B: input tensor of modality B
            x_A: 模态A的输入张量
            x_B: 模态B的输入张量


        Returns:
            A tuple containing the rep_A, rep_B, x_prime_A, and x_prime_B  返回：A和B的表示层，以及A和B的预测输出
        """
        if x_A != None:
            # Forward in the modality A pipe line  在模态A管道中前向传播
            seq_len_A = x_A.shape[1]
            out_A = self.encoder_A(x_A)
            # 获取编码后序列的最后一个时间步的表示，作为重构的表示。
            rep_A = out_A[:, -1,
                          :].unsqueeze(1) if self.batch_first else out_A[-1, :, :].unsqueeze(0)
            #扩充
            rep_seq_A = rep_A.expand(-1, seq_len_A, -1)
            x_prime_A = self.decoder_A(rep_seq_A)

            if x_B == None:
                return(rep_A.squeeze(), None, x_prime_A, None)

        if x_B != None:
            # Forward in the modality B pipe line  在模态A管道中前向传播
            seq_len_B = x_B.shape[1]
            out_B = self.encoder_B(x_B)
            rep_B = out_B[:, -1,
                          :].unsqueeze(1) if self.batch_first else out_B[-1, :, :].unsqueeze(0)
            rep_seq_B = rep_B.expand(-1, seq_len_B, -1)
            x_prime_B = self.decoder_B(rep_seq_B)
            if x_A == None:
                return(None, rep_B.squeeze(), None, x_prime_B)

        return (rep_A.squeeze(), rep_B.squeeze(), x_prime_A, x_prime_B)

    def encode(self, x, modality):
        #通过断言（assertion），确保 modality 的取值为 "A" 或 "B"，如果不是则触发 AssertionError。
        assert (modality == "A" or modality ==
                "B"), "Modality is neither A nor B"
        out = self.encoder_A(x) if modality == "A" else self.encoder_B(x)
        return out


class SplitLSTMAutoEncoder(nn.Module): #
    def __init__(self, input_size_A, input_size_B, representation_size, num_layers=1, batch_first=True):
        super(SplitLSTMAutoEncoder, self).__init__()
        self.batch_first = batch_first
        self.encoder_A = LSTMEncoder(
            input_size=input_size_A, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder_A = LSTMDecoder(representation_size=representation_size,
                                     output_size=input_size_A, num_layers=num_layers, batch_first=batch_first)
        self.encoder_B = LSTMEncoder(
            input_size=input_size_B, representation_size=representation_size, num_layers=num_layers, batch_first=batch_first)
        self.decoder_B = LSTMDecoder(representation_size=representation_size,
                                     output_size=input_size_B, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x, modality):
        #前面基本相同 有两套编解码器
        assert (modality == "A" or modality ==
                "B"), "Modality is neither A nor B"

        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        out = self.encoder_A(x) if modality == "A" else self.encoder_B(x)
        representation = out[:, -1, :].unsqueeze(
            1) if self.batch_first else out[-1, :, :].unsqueeze(0)
        representation_seq = representation.expand(-1, seq_len, -1)
        #重点在这里!!对于一个模态经过编码器输出隐藏表示层（h）,h会分别进入A和B的解码器，得到预测的A和B  （模态之间分离）
        #也就是说他们的目的是要训练出一套可以共享h层的编解码器；也就是从一个模态得到h，那么这个h可以经过两个解码器分别得到对应的A'和B'
        x_prime_A = self.decoder_A(representation_seq)
        x_prime_B = self.decoder_B(representation_seq)
        return (x_prime_A, x_prime_B)

    def encode(self, x, modality):
        assert (modality == "A" or modality ==
                "B"), "Modality is neither A nor B"
        out = self.encoder_A(x) if modality == "A" else self.encoder_B(x)
        return out


class MLP(nn.Module):  #多层感知机模型，用于服务端分类器的训练(只有一个线性层)
    def __init__(self, input_size, n_classes, dropout=0.0):  #定义要传出的参数，输入特征的维度，输出的类别数目。
        super(MLP, self).__init__()
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)  #Dropout 的引入有助于模型更好地泛化到未见过的数据，减少了对训练数据的过度拟合。（在本程序里几乎没有用这个，除了mhealth）
        #！！！Dropout 在每次训练迭代中以一定的概率关闭每个神经元，这样网络在每次迭代中都是不同的子网络
        #这样，每个神经元都不能过于依赖于其他特定的神经元，从而提高模型的鲁棒性。
        #在测试时，所有的神经元都是活动的，但其权重需要按照概率进行缩放，以保持平均活跃度不变。
        self.fc = nn.Linear(input_size, n_classes) #定义一个线性（全连接）层，将输入特征映射到输出类别。

    def forward(self, x):
        out = self.fc(self.dropout(x))
        out = out.contiguous().view(-1, self.n_classes) #将输出张量展平成一个二维张量，其中每行对应一个样本，每列对应一个类别。
        return F.log_softmax(out, dim=1) #将输出张量应用 log_softmax 激活函数，将每个元素的原始输出转换为对应类别的对数概率，做分类器


class ResNetMapper(nn.Module):  #对RGB文件进行特征提取  目的是将输入的图像索引映射为对应的ResNet 512个特征值表示(就是根据feature文件里的匹配帧序号，将RGB与depth图像对齐（特征对齐）)
    resnet = resnet18(pretrained=True).double()
    resnet_mapper = nn.Sequential(*list(resnet.children())[:-1])

    # def map(cls, idxs):
    #     imgs = ur_fall_idxs_to_imgs(idxs)
    #     cls.resnet_mapper.eval()
    #
    #     with torch.no_grad():
    #
    #         x = cls.resnet_mapper(imgs)
    #         x = x.view(x.size(0), -1)
    #
    #     return x
    @classmethod
    def map(cls, idxs, batch_size=32):
        num_samples = len(idxs)
        imgs_list = []

        # 分批处理图像
        for batch_start in range(0, num_samples, batch_size):
            batch_idxs = idxs[batch_start : batch_start + batch_size]
            imgs = ur_fall_idxs_to_imgs(batch_idxs)

            cls.resnet_mapper.eval()

            with torch.no_grad():
                x_batch = cls.resnet_mapper(imgs)
                x_batch = x_batch.view(x_batch.size(0), -1)

            imgs_list.append(x_batch)

        # 将所有批次的结果连接起来
        x = torch.cat(imgs_list, dim=0)

        return x


def process_one(one_file):
    idx_frame, f_img = one_file
    img = Image.open(f_img)
    return (idx_frame, img)


def ur_fall_idxs_to_imgs(idxs):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

    t_imgs = torch.empty(
        (idxs.shape[0], 3, 224, 224), dtype=torch.float64)
    f_list = []
    for idx_frame, frame in enumerate(idxs):
        is_fall = "adl" if frame[0] == 0 else "fall"
        run = int(frame[1])
        frame_num = int(frame[2])
        f_img = os.path.join(r"D:\code_home\ECE535-FederatedLearning-main\data\ur_fall", is_fall, f"cam0-rgb", f"{is_fall}-{str(run).zfill(2)}-cam0-rgb",
                             f"{is_fall}-{str(run).zfill(2)}-cam0-rgb-{str(frame_num).zfill(3)}.png")
        f_list.append((idx_frame, f_img))

    with multiprocessing.Pool(1) as p:
        results = p.map(process_one, f_list)
    for r in results:
        t_imgs[r[0]] = preprocess(r[1]).double()

    return t_imgs
