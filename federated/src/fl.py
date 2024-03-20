import os
import copy
import torch
import math
import numpy as np

from server import Server
from client import Client
from utils import load_data, split_server_train, client_idxs
from pathlib import Path


class FL:
    def __init__(self, config, is_mpi=False, rank=0):
        self.config = config
        self.is_mpi = is_mpi
        self.rank = rank
        self.num_clients_A = int(self.config["FL"]["num_clients_A"])
        self.num_clients_B = int(self.config["FL"]["num_clients_B"])
        self.num_clients_AB = int(self.config["FL"]["num_clients_AB"])
        self.results_path = self.config["SIMULATION"]["results_path"]
        self.rounds = int(self.config["FL"]["rounds"])
        self.eval_interval = int(self.config["FL"]["eval_interval"])

    def start(self):
        """Starts the FL communication rounds between the server and clients."""
        # print current working directory 打印当前工作目录
        print("cwd: ", os.getcwd())
        print("新的试验：")
        # Loads the training and testing data of the FL simumation   载入联邦学习模拟的训练和测试数据
        data_train, data_test = load_data(self.config)

        client_train = data_train  #训练数据集
        server_test = data_test    #测试数据集

        # There is a small chance that the labels in the generated server_train are fewer than the labels in server_test.
        # If that happens, regenerate the server_train again until the sets of lables between them are the same.
        #在每次迭代中都生成新的 server_train_A，并检查其标签集合是否与 server_test 的标签集合相同。如果相同，循环会被中断，否则，将继续生成新的 server_train_A 直到满足条件。
        #标签集合相同应该指的是里面的各类别（0 1 2 3）数量相同
        while True:
            server_train_A = split_server_train(data_train, self.config)
            if set(server_train_A["y"]) == set(server_test["y"]):
                break
        while True:
            server_train_B = split_server_train(data_train, self.config)
            if set(server_train_B["y"]) == set(server_test["y"]):
                break
        #创建服务器实例，并初始化模型；server_train_A、server_train_B 是训练分类器监督学习的数据
        server = Server(server_train_A, server_train_B, self.config)
        #选择模型，并对其初始化
        server.init_models()

        # Generates sample indices for each client
        # 为每个客户端生成样本索引，每个客户端分配的数据量相同（且数据内都包含A、B等全部模态类型数据）
        client_train_idxs = client_idxs(client_train, self.config)
        n_clients = len(client_train_idxs)
        clients = []
        # 按照模态性（Modalities）创建客户端，此部分为加载客户端类型
        modalities = ["A" for _ in range(self.num_clients_A)] + ["B" for _ in range(
            self.num_clients_B)] + ["AB" for _ in range(self.num_clients_AB)]
        #根据模态类型（A/B/AB）创建客户端对象并根据分配的数据索引划分数据，最后将所有客户端添加到 clients 列表中。
        for i in range(n_clients):
            clients.append(
                Client(client_train, client_train_idxs[i], modalities[i], self.config))
       #用于创建结果表格，其中记录了每隔一定轮次进行评估的训练和测试结果
        # n_eval_point计算结果表中的行数，即总轮次除以评估间隔，然后向上取整
        n_eval_point = math.ceil(self.rounds / self.eval_interval)
        # result table: round, local_ae_loss, train_loss, train_accuracy, test_loss, test_accuracy, test_f1
        result_table = np.zeros((n_eval_point, 57))
        #在结果表的第一列中存储轮次的信息
        result_table[:, 0] = np.arange(1, self.rounds+1, self.eval_interval)
        row = 0
        #总体大的训练
        for t in range(self.rounds):
            print(f"Round {t+1} starts")
            #选择客户端（从总数里抽取）
            selected_clients = server.select_clients(clients)
            local_models = []       #里面存储的是 local_model（编码器类型：spilt or dccae）、 client.modality（A、B、AB）、client_weight（α：1or100）

            # Local update on each selected client（一次本地客户端训练得到的更新数据）
            for client in selected_clients:
                local_model, client_weight, local_ae_loss = client.update(
                    copy.deepcopy(server.global_ae))                      #这块就是在交换数据（从服务端->客户端），把服务端全局的模型加载到每个客户端，进行训练
                local_models.append(
                    (copy.deepcopy(local_model), client.modality, client_weight))

            # Cloud update on the server
            train_loss, train_accuracy = server.update(local_models)
            #这块就是在交换数据（客户端->服务端），把客户端训练得到的数据发给服务器，这部分包括多模态聚合算法（全局的编码器融合）、分类器训练，返回值是损失值和准确率

            # Cloud evaluation  每一轮结束后执行的部分，用于进行云端评估，并将评估结果写入结果表
            if t % self.eval_interval != 0:
                continue
            else:
                with torch.no_grad():
                    test_loss, test_accuracy, test_f1, class_occurances_correct, class_occurances_total = server.eval(server_test)


                # print("class_occurances_correct len", len(class_occurances_correct))
                # print(class_occurances_correct)
                # print("class_occurances_total len", len(class_occurances_total))
                # print(class_occurances_total)

                result_table[row] = np.array(
                   (t+1, local_ae_loss, train_loss, train_accuracy, test_loss, test_accuracy, test_f1,  *class_occurances_correct, *class_occurances_total))
                row += 1
                self.write_result(result_table)

    def write_result(self, result_table):
        """ Writes simulation results into a result.txt file

        Args:
            result_table: a 2-d numpy array contraining rows of simulation results
        """
        if self.is_mpi:
            results_path = os.path.join(self.results_path, f"rep_{self.rank}")
        else:
            results_path = self.results_path
        Path(results_path).mkdir(parents=True, exist_ok=True)
        np.savetxt(os.path.join(results_path, "results.txt"),
                   result_table, delimiter=",", fmt="%1.4e")
        