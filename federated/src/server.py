import numpy as np
import torch
import copy

from models import SplitLSTMAutoEncoder, DCCLSTMAutoEncoder, MLP
from torch import nn, optim
from utils import make_seq_batch
from sklearn.metrics import f1_score
#在服务器上，我们对一个带标签的测试数据集进行分类器测试。我们使用一个长度为2,000的滑动时间窗口从测试数据集中提取时间序列序列（不重叠）。
EVAL_WIN = 2000
#自动编码器和分类器都是全局不变的
class Server:
    def __init__(self, train_A, train_B, config):            #都是针对服务器端的（全局模型的更新，分类器的训练），服务器端选好本地训练模型后下发
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_A = train_A
        self.train_B = train_B
        self.input_size_A = train_A["A"].shape[1]                   #A模型数据的特征数量
        self.input_size_B = train_B["B"].shape[1]                   #B模型数据的特征数量
        self.rep_size = int(config["FL"]["rep_size"])               #隐藏层（h）的大小
        self.n_classes = len(set(train_A["y"]))                     #分类的总类别数
        self.label_modality = config["SERVER"]["label_modality"]    #用于分类器训练的有标签数据（从客户端随机抽取出来的）
        self.test_modality = config["SERVER"]["test_modality"]
        self.frac = float(config["SERVER"]["frac"])               #每一轮迭代中从客户端选择参与训练的一部分客户端的比例
        self.n_epochs = int(config["SERVER"]["num_epochs"])       #训练过程中的总迭代轮数的参数
        self.lr = float(config["SERVER"]["lr"])                   #服务器端的学习率
        self.criterion = config["SERVER"]["criterion"]            #通常指定了模型的损失函数
        self.optimizer = config["SERVER"]["optimizer"]            #用于模型参数优化的优化算法。优化算法的目标是通过调整模型参数，使损失函数达到最小值。
        self.model_ae = config["SIMULATION"]["model_ae"]          #这个参数用于选择自编码器
        self.model_sv = config["SIMULATION"]["model_sv"]          #这个参数用于选择支持向量机->分类器的监督学习方式MLP）
        self.mlp_dropout = 0.5 if config["SIMULATION"]["data"] == "mhealth" else 0.0  #MLP中的遗忘率

        if config["SIMULATION"]["data"] == "ur_fall":
            self.batch_min = 16
            self.batch_max = 32
        else:
            self.batch_min = 128
            self.batch_max = 256

    def init_models(self):  #全局模型的设定，model_ae是从两个当中选一个当做编码器，model_sv就是MLP训练分类器
        if self.model_ae == "split_LSTM":
            self.global_ae = SplitLSTMAutoEncoder(
                self.input_size_A, self.input_size_B, self.rep_size).double().to(self.device)

        if self.model_ae == "DCCAE_LSTM":
            self.global_ae = DCCLSTMAutoEncoder(
                self.input_size_A, self.input_size_B, self.rep_size).double().to(self.device)

        if self.model_sv == "MLP":  #分类器MLP
            self.global_sv = MLP(
                self.rep_size, self.n_classes, self.mlp_dropout).double().to(self.device)

    def select_clients(self, clients):  #根据比例，从总的客户端里选择对应比例的客户端出来进行本地迭代训练
        """Selects clients to communicate with.  从给定的客户端列表中随机选择一部分客户端进行通信

        Args:
            clients: a list of Client objects

        Returns:
            A list of selected Client objects
        """
        n_selected_clients = int(len(clients) * self.frac)
        selected_clients = np.random.choice(
            clients, n_selected_clients, replace=False)
        return selected_clients

    def average_models(self, local_models):    #与fedavg相比应该就增加了一个α权重
        """Averages local models into a new global model.

        Args:
            local_models: a list of tuples containing models, client modalities, and client weights
                          #每个元组中包含了本地模型、客户端模态和客户端权重
            元组的结构为 (model, modality, weight)，其中：
            model: 本地模型对象，可能是自编码器、支持向量机等。
            modality: 字符串，表示客户端的模态，可能是 "A"、"B" 或 "AB"。
            weight: 浮点数，表示客户端的权重，用于加权平均模型。，就是阿尔法1（A/B） 100（AB）
        Returns:
            A new global model.
        """
        w_avg_A = None
        w_avg_B = None
        n_A = 0
        n_B = 0
        for model in local_models:
            if model[1] == "A" or model[1] == "AB":           # modality
                n_A += model[2]                               #在计算A模型的
                if not w_avg_A:
                    w_avg_A = copy.deepcopy(model[0].state_dict())   #提取模型参数（返回的是一个字典）
                    for key in w_avg_A.keys():
                        if "A" in key:
                            w_avg_A[key] = w_avg_A[key] * model[2]  #模型每个参数都乘权重（1或100）
                else:
                    for key in w_avg_A.keys():
                        if "A" in key:
                            # multiply client weight  #这个语法中的反斜杠 \ 用于将一行代码拆分成两行，以提高代码的可读性
                            w_avg_A[key] += model[0].state_dict()[key] * \
                                model[2]

            if model[1] == "B" or model[1] == "AB":
                n_B += model[2]
                if not w_avg_B:
                    w_avg_B = copy.deepcopy(model[0].state_dict())
                    for key in w_avg_B.keys():
                        if "B" in key:
                            w_avg_B[key] = w_avg_B[key] * model[2]
                else:
                    for key in w_avg_B.keys():
                        if "B" in key:
                            w_avg_B[key] += model[0].state_dict()[key] * \
                                model[2]
        w_avg = w_avg_A if w_avg_A else w_avg_B

        if w_avg_A:
            for key in w_avg.keys():
                if "A" in key:
                    w_avg[key] = w_avg_A[key] / n_A
        if w_avg_B:
            for key in w_avg.keys():
                if "B" in key:
                    w_avg[key] = w_avg_B[key] / n_B

        return w_avg

    def train_classifier(self, label_modality, optimizer, criterion, x_train, y_train, idx_start, idx_end):
        """Trains the global classifier with labelled data on the server"""

        #从 x_train 中提取指定范围的序列数据，并将其转换为 PyTorch 张量 seq，类型为 double，并移动到设备（GPU 或 CPU）上。
        x = x_train[:, idx_start:idx_end, :]
        seq = torch.from_numpy(x).double().to(self.device)
        y = y_train[:, idx_start:idx_end]

        with torch.no_grad():
            rpts = self.global_ae.encode(seq, label_modality)     #rtps = 上一步全局编码器模型输出的隐藏层表示
        targets = torch.from_numpy(y.flatten()).to(self.device)   #目标值

        optimizer.zero_grad()
        output = self.global_sv(rpts)
        loss = criterion(output, targets.long()) #损失函数计算损失值
        top_p, top_class = output.topk(1, dim=1) # 从output每行中找到最大的一个元素。
        #top_p 包含了每行最大元素的概率值  top_class 包含了对应于最大概率的类别索引
        equals = top_class == targets.view(*top_class.shape).long()
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
         #返回当前批次的损失值和准确率
        return loss.item(), accuracy

    def update(self, local_models):   #使用multi-fedavg算法得到更新的全局编码器参数权重（一次），之后训练分类器（数次，n_epochs）->主要就是这个
        """Updates the global model using received local models.

        Args:
            local_models: a list of local models #客户端的一系列模型列表

        Returns:
            A tuple containing loss and accuracy values    返回的是：包含损失和准确率值的元组
        """
        # Average all local models and update the global ae
        global_weights = self.average_models(local_models)  #使用多模态fedavg算出模型的权重
        self.global_ae.load_state_dict(global_weights)
        self.global_ae.eval()                     ## 设置为评估模式，不进行梯度计算
        self.global_sv.train()                    # 设置为训练模式，进行梯度计算

        if self.criterion == "CrossEntropyLoss":    #分类器训练的损失函数（所有都一样）
            criterion = nn.CrossEntropyLoss().to(self.device)
        if self.optimizer == "Adam":                #分类器训练的优化器（所有都一样）
            optimizer = optim.Adam(self.global_sv.parameters(), lr=self.lr)

        round_loss = []
        round_accuracy = []
        for epoch in range(self.n_epochs):
            epoch_loss = []
            epoch_accuracy = []
            #选择batch的大小
            batch_size = np.random.randint(
                low=self.batch_min, high=self.batch_max)
            # 从模态 "A" 中获取训练数据
            x_A_train, _, y_A_train = make_seq_batch(
                self.train_A, [0], len(self.train_A["A"]), batch_size)
            # 从模态 "B" 中获取训练数据
            _, x_B_train, y_B_train = make_seq_batch(
                self.train_B, [0], len(self.train_B["B"]), batch_size)
            # A_train and B_train both are in the shape of (batch_size, seq_len, input_size), i.e., batch first

            # 针对模态 "A" 进行训练（分类器得到的标签数据），分批喂数据进去
            if "A" in self.label_modality:
                seq_len = x_A_train.shape[1]
                idx_start = 0
                idx_end = 0
                while idx_end < seq_len:
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)
                    loss, accuracy = self.train_classifier(
                        "A", optimizer, criterion, x_A_train, y_A_train, idx_start, idx_end)
                    epoch_loss.append(loss)
                    epoch_accuracy.append(accuracy)

            if "B" in self.label_modality:
                seq_len = x_B_train.shape[1]
                idx_start = 0
                idx_end = 0
                while idx_end < seq_len:
                    win_len = np.random.randint(low=16, high=32)
                    idx_start = idx_end
                    idx_end += win_len
                    idx_end = min(idx_end, seq_len)
                    loss, accuracy = self.train_classifier(
                        "B", optimizer, criterion, x_B_train, y_B_train, idx_start, idx_end)
                    epoch_loss.append(loss)
                    epoch_accuracy.append(accuracy)

            round_loss.append(np.mean(epoch_loss))
            round_accuracy.append(np.mean(epoch_accuracy))

        return np.mean(round_loss), np.mean(round_accuracy)

    def eval(self, data_test):
        #评估全局模型在测试数据上的性能，并返回损失、准确率、加权 F1 分数以及每个类别的正确预测次数和总预测次数。
        """Evaluates global models against testing data on the server.    评估服务器上的全局模型对测试数据的性能

        Args:
            data_test: a dictionary containing testing data of modalities A&B and labels y. 包含模态 A 和 B 以及标签 y 的测试数据的字典。

        Returns:
            A tuple containing loss and accuracy values  包含损失和准确率值的元组
        """
        self.global_ae.eval()
        self.global_sv.eval()
        if self.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss().to(self.device)

        if self.test_modality == "A":
            x_samples = np.expand_dims(data_test["A"], axis=0)
        elif self.test_modality == "B":
            x_samples = np.expand_dims(data_test["B"], axis=0)
        y_samples = np.expand_dims(data_test["y"], axis=0)

        win_loss = []
        win_accuracy = []
        win_f1 = []
        n_samples = x_samples.shape[1]
        n_eval_process = n_samples // EVAL_WIN + 1

        # np_gt is a numpy.ndarray of labels (classes)
        # equals is a tensor of booleans representing if the model correctly guessed 
        class_occurances_total = [0] * 25
        class_occurances_correct = [0] * 25

        for i in range(n_eval_process):
            idx_start = i * EVAL_WIN
            idx_end = np.min((n_samples, idx_start+EVAL_WIN))
            x = x_samples[:, idx_start:idx_end, :]
            y = y_samples[:, idx_start:idx_end]
        
            inputs = torch.from_numpy(x).double().to(self.device)
            targets = torch.from_numpy(y.flatten()).to(self.device)
            rpts = self.global_ae.encode(inputs, self.test_modality)
            output = self.global_sv(rpts)

            loss = criterion(output, targets.long())
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            np_gt = y.flatten()
            np_pred = top_class.squeeze().cpu().detach().numpy()
            weighted_f1 = f1_score(np_gt, np_pred, average="weighted")

            win_loss.append(loss.item())
            win_accuracy.append(accuracy)
            win_f1.append(weighted_f1)

            # count the number of times each class was predicted correctly
            # loop through each prediction
            for i in range(len(np_gt)):
                #print(len(np_gt))
                class_index = np_gt[i]
                #print("print(class_index)", class_index)
                # increment total occurances for that class
                class_occurances_total[int(class_index)] += 1
                # if the prediction was correct, increment correct occurances for that class
                if equals[i] == True:
                    class_occurances_correct[int(class_index)] += 1

            # print("class_occurances_correct len", len(class_occurances_correct))
            # print(class_occurances_correct)

            # print("\nclass_occurances_total len", len(class_occurances_total))
            # print(class_occurances_total)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        
       
        return np.mean(win_loss), np.mean(win_accuracy), np.mean(win_f1), class_occurances_correct, class_occurances_total
