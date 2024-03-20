import os
import requests
import numpy as np
import pandas as pd
import torch
import datetime

from scipy.io import savemat, loadmat
from scipy.stats import zscore
from models import ResNetMapper

N_DIV_OPP = 100
N_DIV_MHEALTH = 100
N_DIV_URFALL = 10
N_LABEL_DIV_OPP = 15
N_LABEL_DIV_MHEALTH = 9
N_LABEL_DIV_URFALL = 9


def fill_nan(matrix):
    """Fill NaN values with the value of the same column from previous row #对值为空（动作）的数据，进行处理，替换为相邻行的对应元素值

    Args:
        matrix: a 2-d numpy matrix  一个二维矩阵！
    Return:
        A 2-d numpy matrix with NaN values filled 填充完的Nan数据
    """
    m = matrix
    np.nan_to_num(x=m[0, :], copy=False, nan=0.0)  #将第一行中的NaN值替换为0.0，以确保后续对第一行的处理不会出现NaN值。
    for row in range(1, m.shape[0]):               #遍历矩阵的每一行(第二行开始)
        for col in range(m.shape[1]):              #遍历矩阵的每一列
            if np.isnan(m[row, col]):              #如果当前元素是NaN，用相邻行的对应元素的值进行替换
                m[row, col] = m[row-1, col]
    return m


def gen_mhealth(data_path):
    """Generates subjects' data in .mat format from the mHealth dataset. 将处理完的数据生成.mat格式

    The experiments on the mHealth dataset are done in the fashion of leave-one-subject-off.  !!!mHealth数据集上的实验采用一种留一主体法的方式进行。
    So the .mat data is indexed by subjects instead of "training", "validating", and "testing". .mat数据是按主体索引的，而不是按“training”、“validating”和“testing”划分的。

    Args:
        data_path: the path of the mHealth dataset.

    Returns:
        None
    """
    # 为加速度、陀螺仪、磁力计和标签定义列索引
    acce_columns = [0, 1, 2, 5, 6, 7, 14, 15, 16]
    gyro_columns = [8, 9, 10, 17, 18, 19]
    mage_columns = [11, 12, 13, 20, 21, 22]
    y_column = 23


    # 创建一个字典来存储数据
    mdic = {}
    # 创建一个集合来存储唯一的标签
    labels = set()
    # 创建一个列表来存储每个参与者数据的形状
    shape_list = []

    for i in range(1, 11):   #10个数据集都处理（用的是 leave-one-subject-off方法）
        # 加载当前参与者的数据
        s_data = np.loadtxt(os.path.join(
            data_path, "mhealth", f"mHealth_subject{i}.log"))
        # 提取数据列并处理NaN值
        x_acce = fill_nan(s_data[:, acce_columns])
        x_gyro = fill_nan(s_data[:, gyro_columns])
        x_mage = fill_nan(s_data[:, mage_columns])
        y = s_data[:, y_column]
        # 使用参与者特定的键将数据存储在字典中
        mdic[f"s{i}_acce"] = x_acce
        mdic[f"s{i}_gyro"] = x_gyro
        mdic[f"s{i}_mage"] = x_mage
        mdic[f"s{i}_y"] = y
        # 更新唯一标签的集合
        labels = labels.union(set(y))
        # 打印参与者数据的形状
        print(f"shape of participant {i}: {s_data.shape}")
        shape_list.append(s_data.shape[0])

    # 打印数据形状的平均值和标准差
    print(f"mean:{np.mean(shape_list)}, std:{np.std(shape_list)}")
    # 将唯一标签的集合转换为排序后的列表，准备做标签的映射
    unique_y = list(labels)
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
        # 使用映射更新字典中原本的标签，
    for i in range(1, 11):
        mdic[f"s{i}_y"] = np.squeeze(
            np.vectorize(y_map.get)(mdic[f"s{i}_y"]))
     #以.mat文件保存
    savemat(os.path.join(data_path, "mhealth", "mhealth.mat"), mdic)


def gen_opp(data_path): #(该数据集都是 数字序列 )
    """Generates training, validating, and testing data from Opp datasets

    Args:
        data_path: the path of the Opportunity challenge dataset

    Returns:
        None
    """
    #定义加速度和陀螺仪的列索引，对应相关的数据
    acce_columns = [i-1 for i in range(2, 41)]
    acce_columns.extend([46, 47, 48, 55, 56, 57, 64, 65, 66, 73, 74,
                         75, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105, 106, ])
    gyro_columns = [40, 41, 42, 49, 50, 51,
                    58, 59, 60, 67, 68, 69, 66, 67, 68, ]
    # Loads the run 2 from subject 1 as validating data 作为一个验证数据集
    data_valid = np.loadtxt(os.path.join(data_path, "opp", "S1-ADL2.dat"))
    x_valid_acce = fill_nan(data_valid[:, acce_columns])             #加速度数据对NaN值处理
    x_valid_gyro = fill_nan(data_valid[:, gyro_columns])             #陀螺仪数据对NaN值处理
    y_valid = data_valid[:, 115]                                     #标签数据（第115个是）动作标签

    # Loads the runs 4 and 5 from subjects 2 and 3 as testing data 处理 测试模型的数据集，ADL4和ADL5的2,3数据集
    runs_test = []                             #run_test是一个有文件路径和测试数据集的名（路径+'opp'+S2-ADL4）
    idxs_test = []                            #idxs_tedt，是一个类似于索引数据集的并每个元素是元组（4,2）、（4,3）....
    for r in [4, 5]:
        for s in [2, 3]:
            runs_test.append(np.loadtxt(os.path.join(
                data_path, "opp", f"S{s}-ADL{r}.dat")))
            idxs_test.append((r, s))
    data_test = np.concatenate(runs_test)                   #将多个数据文件的内容合并成一个大的数据集
    x_test_acce = fill_nan(data_test[:, acce_columns])        #加速度数据对NaN值处理
    x_test_gyro = fill_nan(data_test[:, gyro_columns])        #陀螺仪数据对NaN值处理
    y_test = data_test[:, 115]                                #标签数据（第115个是）动作标签

    # Loads the remaining runs as training data        处理 训练模型的数据集
    runs_train = []
    for r in range(1, 6):
        for s in range(1, 5):
            if (r, s) not in idxs_test:                       #排除上面用过的测试数据集
                runs_train.append(np.loadtxt(os.path.join(
                    data_path, "opp", f"S{s}-ADL{r}.dat")))
    data_train = np.concatenate(runs_train)                   # #将多个数据文件的内容合并成一个大的数据集
    x_train_acce = fill_nan(data_train[:, acce_columns])      #处理NaN值
    x_train_gyro = fill_nan(data_train[:, gyro_columns])
    y_train = data_train[:, 115]                             #提取标签

    # Changes labels to (0, 1, ...)  将标签（y_XXX）从原始的类别标签映射到以0为起始的连续整数标签
    unique_y = list(set(y_train).union(set(y_valid)).union(set(y_test))) #创建一个包含所有训练、验证和测试集中唯一标签的列表（利用集合的不重复性，并把所有标签都集中）
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx                              # 构建标签映射字典（因为原来的标签很长 现在变成0 1 2 这种）
    y_train = np.vectorize(y_map.get)(y_train)     # 使用映射字典转换训练标签
    y_valid = np.vectorize(y_map.get)(y_valid)     # 使用映射字典转换验证标签
    y_test = np.vectorize(y_map.get)(y_test)       # 使用映射字典转换测试标签

    mdic = {}                                    #用字典mdic存储数据 将原来的只有数字的数据集变成了字典模式有对应内容的数据集
    mdic["x_train_acce"] = x_train_acce
    mdic["x_train_gyro"] = x_train_gyro
    mdic["y_train"] = np.squeeze(y_train)
    mdic["x_valid_acce"] = x_valid_acce
    mdic["x_valid_gyro"] = x_valid_gyro
    mdic["y_valid"] = np.squeeze(y_valid)  # This only has 17 classes
    mdic["x_test_acce"] = x_test_acce
    mdic["x_test_gyro"] = x_test_gyro
    mdic["y_test"] = np.squeeze(y_test)

    savemat(os.path.join(data_path, "opp", "opp.mat"), mdic)  #保存为.mat格式


def gen_ur_fall(data_path):  #为UR_Fall数据集生成训练和测试数据，此数据集的训练和测试数据也是抽取得来的
    #总共70个（30+40），每次抽1/10（即，7个）作为测试的，其余63个作为训练集。每个客户端随机抽样的序列大小是总训练数据的1/9（联邦学习中划分训练集）
    """Generates training and testing data for UR Fall datasets.

    Args:
        data_path: the path of the UR Fall datasets.

    Returns:
        None
    """
    # headers
    # fall (0 or 1), run (1-40 for fall=0, 1-30 for fall=1), frame, HeightWidthRatio, MajorMinorRatio, BoundingBoxOccupancy, MaxStdXZ, HHmaxRatio, H, D, P40, acce_x, acce_y, acce_z, y
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    a_list = []
    runs = [40, 30]  # ADL（40）摔倒（30）和的运行序列数目
    shape_list = []
    a_row=0
    for fall in range(2): # 对于摔倒（fall=1）和ADL（fall=0）
        prefix = "fall" if fall == 1 else "adl"
        f_labelled = os.path.join(                                               #os.path.join 函数将路径的各个部分连接起来，形成完整的文件路径。
            data_path, "ur_fall", prefix, f"urfall-features-cam0-{prefix}s.csv") #返回值 f_labelled 是一个字符串，代表了深度图像特征文件的完整路径。
        df_labelled = pd.read_csv(                                               #pd.read_csv 函数读取深度图像特征文件（由 f_labelled 指定）的内容
            f_labelled, delimiter=",", header=None, usecols=list(range(11)))     #最终，df_labelled 是一个包含深度图像特征数据的 DataFrame。
        #此部分就是读取下载的对于depth数据进行处理的相关提取其特征的文件（urfall-features-cam0-falls，urfall-features-cam0-adls）

        for run in range(1, runs[fall]+1):
            f_acc = os.path.join(data_path, "ur_fall", prefix,
                                 "acc", f"{prefix}-{str(run).zfill(2)}-acc.csv")  #加速度数据文件的路径（excel文件）
            f_sync = os.path.join(
                data_path, "ur_fall", prefix, "sync", f"{prefix}-{str(run).zfill(2)}-data.csv") #syn同步数据的文件路径（excel文件）
            #为了做到每个视频帧都与加速度计数据匹配，同步数据包括：帧编号，距离序列开始的时间，插值加速度数据，对应图像帧
            data_acce = np.genfromtxt(f_acc, delimiter=",")
            data_sync = np.genfromtxt(f_sync, delimiter=",") #同步数据的第一列：帧编号；第二列：鼓励序列开始的毫秒数；第三列：插值加速度数据数据（adL中没有这个）
            df_label_part = df_labelled[df_labelled[0]
                                        == f"{prefix}-{str(run).zfill(2)}"]    #分步提取文件urfall-features-cam0-falls，urfall-features-cam0-adls中adl-001（002...）等数据
            n_rows = df_label_part.shape[0]               #获取选择后的数据框 df_label_part 的行数，存储在变量 n_rows 中
            a = np.zeros([n_rows, 15])                    #创建一个大小为 (n_rows, 15) 的全零矩阵，用于存储处理后的数据。
            print(n_rows)
            a[:, 0] = fall                                #a的第一列是 fall（0或1的数值）；对于摔倒（fall=1）和ADL（fall=0
            a[:, 1] = run                                 #a的第二列是run(40或30)；表示ADL和fall中第几个数据（001、002...）
            a[:, 2] = df_label_part[1].to_numpy()         #将矩阵 a 的第三列设为 df_label_part 中（feaatures文件）第二列的值。帧编号-对应于序列中的编号
            a[:, 3:11] = df_label_part[df_label_part.columns.intersection(
                list(range(3, 11)))].to_numpy()           #将矩阵 a 的第4到第11列设为 df_label_part 中第4到第11列的值（其他数据的值）。
            a[:, 14] = df_label_part[2].to_numpy()
            # 将矩阵 a 的最后一列设为 df_label_part 中第三列的值。标签-在深度框中描述人体姿态;“-1”表示人没有躺下，“1”表示人躺在地上;“0”是暂时的姿势，当人“正在坠落”时，我们在分类时不使用“0”帧，
            mask = [x in a[:, 2] for x in data_sync[:, 0]] #找到相同的帧编号导出（a的第三列和syn中的第一列）
            timestamps = data_sync[mask, 1]                #找到对应的时间（此序列至开始时刻的时间ms）
            acce_xyz = np.empty((0, 3), dtype=np.float64)
            row_acce_data = 0
            #上面这部分代码主要是将深度图像的特征、运行信息、同步数据和加速度数据整合到一个数组中，以便后续处理和分析。
            #下面这段代码的目的是将加速度数据匹配到深度图像特征数据，
            for ts in timestamps:
                while row_acce_data < data_acce.shape[0] and data_acce[row_acce_data, 0] < ts:
                    row_acce_data += 1
                if row_acce_data >= data_acce.shape[0]:
                    break
                if abs(data_acce[row_acce_data, 0] - ts) < abs(data_acce[row_acce_data-1, 0] - ts):
                    acce_xyz = np.append(
                        acce_xyz, [data_acce[row_acce_data, 2:5]], axis=0)
                else:
                    acce_xyz = np.append(
                        acce_xyz, [data_acce[row_acce_data-1, 2:5]], axis=0)
            if acce_xyz.shape[0] < a.shape[0]:
                n = a.shape[0] - acce_xyz.shape[0]
                a = a[:-n, :]
            a[:, 11:14] = acce_xyz  #上面的循环就是处理加速度数据，使其与视频帧匹配，一个活动是一个a（001 002 003....）这确保了每个视频帧都与相应的加速度数据对应。
            a_list.append(a)        #a_list是把所有的a聚集
            shape_list.append(a.shape[0])  #shape_list记录的是每个a的行数
            print(f"shape: {a.shape}")
            a_row+=a.shape[0]
    print(f"mean:{np.mean(shape_list)}, std:{np.std(shape_list)}")
    print(f"length of a_list:{len(a_list)}")
    data = np.concatenate(a_list) #将a_list中70个a合并成一个大的数据集a（x*15）
    print(f"data.shape:{data.shape}")
    print(f"a_row:{a_row}")

    #这段代码的目的是将深度图像特征数据、加速度数据、RGB 图像特征数据和相关的索引和标签整合到一个字典中，并将字典保存为 MATLAB 格式的文件
    #将深度图像特征数据（depth）、加速度数据（acce）、标签数据（y）、RGB图像特征数据（rgb）整合到一个字典（mdic）中。
    mdic = {}
    mdic["depth"] = data[:, 0:11]
    mdic["acce"] = data[:, [0, 1, 2, 11, 12, 13]]
    mdic["y"] = data[:, [0, 1, 2, 14]]
    idxs_rgb = data[:, [0, 1, 2]]                       #索引（idxs_rgb）用于将RGB图像特征与深度图像特征关联起来，对应配对。
    rgb_features = ResNetMapper.map(idxs_rgb).numpy()
    mdic["rgb"] = np.empty((data.shape[0], rgb_features.shape[1]+3))
    mdic["rgb"][:, [0, 1, 2]] = idxs_rgb
    mdic["rgb"][:, range(3, rgb_features.shape[1]+3)] = rgb_features
    #对y值（最后一列是标签）做映射
    y_old = data[:, 14]
    unique_y = list(set(y_old))
    unique_y.sort()

    y_map = {}
    for idx, y in enumerate(unique_y):
        y_map[y] = idx
    mdic["y"][:, 3] = np.vectorize(y_map.get)(y_old)

    savemat(os.path.join(data_path, "ur_fall", "ur_fall.mat"), mdic)


def load_data(config):     #返回的是根据数据类型分类的不同模态数据，例：opp中（modality_A->acce;modality_B->gyro）
    """Loads the dataset of the FL simulation.


    Args:
        config: a map of configurations of the simulation ;config是一个字典

    Returns:
        A dictionary containing training and testing data for modality A&B and labels.
    """

    data = config["SIMULATION"]["data"]
    data_path = config["SIMULATION"]["data_path"]  #注意要找到这个data_path在那
    modality_A = config["SIMULATION"]["modality_A"]
    modality_B = config["SIMULATION"]["modality_B"]

    if data == "opp":
        modalities = ["acce", "gyro"]     #定义了一个包含有效模态的列表 modalities，在这个例子中，包括 "acce" 和 "gyro"。
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is neither acce nor gyro."
           #使用断言（assert）确保配置中指定的模态 A 和模态 B 都在有效的模态列表中，如果不是则抛出一个异常，提示模态既不是 "acce" 也不是 "gyro"。
        mat_data = loadmat(os.path.join(data_path, "opp", "opp.mat"))   #把前面经过预处理得到的mat文件路径传过来

        data_train = {"A": zscore(mat_data[f"x_train_{modality_A}"]), "B": zscore(
            mat_data[f"x_train_{modality_B}"]), "y": np.squeeze(mat_data["y_train"])} #获取训练数据集（modality_A->acce;modality_B->gyro）
        data_test = {"A": zscore(mat_data[f"x_test_{modality_A}"]), "B": zscore(
            mat_data[f"x_test_{modality_B}"]), "y": np.squeeze(mat_data["y_test"])}
        return (data_train, data_test)
    elif data == "mhealth":
        modalities = ["acce", "gyro", "mage"]
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is not acce, gyro, or mage."
        mat_data = loadmat(os.path.join(data_path, "mhealth", "mhealth.mat"))
        # Randomely chooses 1 subject among all 10 subjects as testing data and the rest as training data
        s_test = np.random.randint(1, 11)  #随机选择一个主体作为测试数据，剩余的9个作为训练数据。
        data_train = {"A": [], "B": [], "y": []}     #设置测试（1个）、训练（9个，最后要合并为一个大的字典）的数据集
        data_test = {}
        for i in range(1, 11):
            if i == s_test:
                data_test["A"] = zscore(mat_data[f"s{i}_{modality_A}"])
                data_test["B"] = zscore(mat_data[f"s{i}_{modality_B}"])
                data_test["y"] = np.squeeze(mat_data[f"s{i}_y"])
            else:
                data_train["A"].append(zscore(mat_data[f"s{i}_{modality_A}"]))
                data_train["B"].append(zscore(mat_data[f"s{i}_{modality_B}"]))
                data_train["y"].append(mat_data[f"s{i}_y"])
        data_train["A"] = np.concatenate(data_train["A"])
        data_train["B"] = np.concatenate(data_train["B"])
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"], axis=1))
        return (data_train, data_test)
    elif data == "ur_fall":
        modalities = ["acce", "rgb", "depth"]             #确定模型类型种类
        assert (
            modality_A in modalities and modality_B in modalities), "Modality is not acce, rgb, or depth."
        mat_data = loadmat(os.path.join(data_path, "ur_fall", "ur_fall.mat"))    #文件路径
        fall_test = np.random.choice(range(1, 31), 3, replace=False) #随机选择3个作为测试集（对fall）,即抽取10%
        adl_test = np.random.choice(range(1, 41), 4, replace=False)  #随机选择4个作为测试集（对adl）,即抽取10%
        data_train = {"A": [], "B": [], "y": []}
        data_test = {"A": [], "B": [], "y": []}
        #对于每个样本，根据模态类型对数据进行预处理，如果是 "acce" 或 "depth" 类型，还对数据进行 z-score 标准化。
        a_A = mat_data[modality_A]
        a_B = mat_data[modality_B]
        a_y = mat_data["y"]

        for i in range(1, 31):
            sub_a_A = a_A[(a_A[:, 0] == 1) & (a_A[:, 1] == i), :] #sub_a_A 包含了在 a_A 数组中，第一列等于1且第二列等于 i 的所有行。(看那个data各列数据说明图片)
            sub_a_B = a_B[(a_B[:, 0] == 1) & (a_B[:, 1] == i), :]#也就是在做选准数据集的操作
            sub_a_y = a_y[(a_y[:, 0] == 1) & (a_y[:, 1] == i), :]
            if modality_A == "acce" or modality_A == "depth":
                sub_a_A[:, 3:] = zscore(sub_a_A[:, 3:]) #对第3列之后（前三列都是一样的对齐信息而已，没有用的数据）的数据做zscore操作（具体是干什么的看代码学习文件，就是一个数据处理的方法）
            if modality_B == "acce" or modality_B == "depth":
                sub_a_B[:, 3:] = zscore(sub_a_B[:, 3:])

            sub_a_A = sub_a_A[:, 3:]   #去掉前三列
            sub_a_B = sub_a_B[:, 3:]
            sub_a_y = sub_a_y[:, 3]

            if i in fall_test:
                data_test["A"].append(sub_a_A)
                data_test["B"].append(sub_a_B)
                data_test["y"].append(sub_a_y)
            else:
                data_train["A"].append(sub_a_A)
                data_train["B"].append(sub_a_B)
                data_train["y"].append(sub_a_y)

        for i in range(1, 41):
            sub_a_A = a_A[(a_A[:, 0] == 0) & (a_A[:, 1] == i), :]
            sub_a_B = a_B[(a_B[:, 0] == 0) & (a_B[:, 1] == i), :]
            sub_a_y = a_y[(a_y[:, 0] == 0) & (a_y[:, 1] == i), :]
            if modality_A == "acce" or modality_A == "depth":
                sub_a_A[:, 3:] = zscore(sub_a_A[:, 3:])
            if modality_B == "acce" or modality_B == "depth":
                sub_a_B[:, 3:] = zscore(sub_a_B[:, 3:])

            sub_a_A = sub_a_A[:, 3:]
            sub_a_B = sub_a_B[:, 3:]
            sub_a_y = sub_a_y[:, 3]

            if i in adl_test:
                data_test["A"].append(sub_a_A)
                data_test["B"].append(sub_a_B)
                data_test["y"].append(sub_a_y)
            else:
                data_train["A"].append(sub_a_A)
                data_train["B"].append(sub_a_B)
                data_train["y"].append(sub_a_y)

        data_train["A"] = np.concatenate(data_train["A"])
        data_train["B"] = np.concatenate(data_train["B"])
        data_train["y"] = np.squeeze(np.concatenate(data_train["y"]))
        data_test["A"] = np.concatenate(data_test["A"])
        data_test["B"] = np.concatenate(data_test["B"])
        data_test["y"] = np.squeeze(np.concatenate(data_test["y"]))
        return (data_train,  data_test)


def split_server_train(data_train: object, config: object) -> object:
    """Extracts training data for the server.   从训练数据中提取用于服务器训练的数据。上传的数据是随机选择得到的一个客户端内的所有数据

    Args:
        data_train: a dictionary of training data of modalities A&B and labels y 包含模态 A、B 和标签 y 的训练数据的字典
        config: a map of configurations of the simulation  模拟配置的字典

    Returns:
    A dictionary containing the server training data.  包含服务器训练数据的字典。
    """
    #  从配置中获取训练数据的比例
    train_supervised_ratio = float(config["FL"]["train_supervised_ratio"])
    # 获取模态 A、B 和标签 y 的训练数据
    x_train_A = data_train["A"]
    x_train_B = data_train["B"]
    y_train = data_train["y"]
    # 初始化服务器训练数据的空数组
    server_train_A = np.empty((0, x_train_A.shape[1]))
    server_train_B = np.empty((0, x_train_B.shape[1]))
    server_train_y = np.empty((0))

    # 根据数据集类型选择标签的分割数
    if config["SIMULATION"]["data"] == "opp":
        n_div = N_LABEL_DIV_OPP   #（15，从opp的大的测试集里（由若干个数据集合成的一个大数据库）抽取数据，1/15）
    elif config["SIMULATION"]["data"] == "mhealth":
        n_div = N_LABEL_DIV_MHEALTH  #（9）
    elif config["SIMULATION"]["data"] == "ur_fall":
        n_div = N_LABEL_DIV_URFALL  #（9）
    # 计算服务器训练样本的数量
    n_server_train = round(n_div * train_supervised_ratio)
    # 计算每个标签分割的样本数
    n_row = len(x_train_A) #训练数据的大数据库里面的数据个数
    n_sample_per_div = n_row // n_div  #总数据个数/分割数，一个客户端分配到的数据量

    idxs = np.arange(0, n_row, n_sample_per_div) #创建一个数组，包含从0到n_row-1的索引，步长为n_sample_per_div

    slices_A = np.split(x_train_A, idxs) #得到的是将一个大的数据库里的所有数据进行分片，也就是一个客户端获得（分配）的数据
    #np.split(array, indices)
    #分割的规则是：从索引 indices[0] 开始，到 indices[1] 之前为一组，然后从 indices[1] 开始，到 indices[2] 之前为一组，以此类推。
    slices_B = np.split(x_train_B, idxs)
    slices_y = np.split(y_train, idxs)
    del slices_A[0] #删除第一个元素（删除第一个子数组）也就是剔除了
    del slices_B[0]
    del slices_y[0]
    n_slices = len(slices_A) #刨去第一个分片后，剩余分片的数量
    idxs_server_train = np.random.choice(                    #也就是从客户端里随机选择n_server_train个上传数据到服务器
        np.arange(n_slices), n_server_train, replace=False)  #设置服务器监督学习的idx,从 np.arange(n_slices) 数组中随机选择 n_server_train个索引
    #按照索引添加数据
    for i in range(n_slices):
        if i in idxs_server_train:
            server_train_A = np.concatenate((server_train_A, slices_A[i]))
            server_train_B = np.concatenate((server_train_B, slices_B[i]))
            server_train_y = np.concatenate((server_train_y, slices_y[i]))
    server_train = {"A": server_train_A,
                    "B": server_train_B, "y": server_train_y}
    return server_train


def get_seg_len(n_samples, config):  #根据数据集类型和训练比例计算每个分割的样本数。n_samples: 总样本数
    if config["SIMULATION"]["data"] == "opp":
        n_div = N_DIV_OPP  #100
    elif config["SIMULATION"]["data"] == "mhealth":
        n_div = N_DIV_MHEALTH #100
    elif config["SIMULATION"]["data"] == "ur_fall":
        n_div = N_DIV_URFALL #10
    return int(n_samples * float(config["FL"]["train_ratio"])//n_div)


def make_seq_batch(dataset: object, seg_idxs: object, seg_len: object, batch_size: object) -> object:   #就是将一个大数据根据batch_size大小划分为小的数据集
    """Makes batches of sequences from the dataset.
    根据给定的数据集和参数生成一个包含多个序列批次的元组。生成的批次包括模态A、B的序列和相应的标签y。

    Args:
        dataset: a dictionary containing data of modalities A&B and labels y
        seg_idxs: A list containing the starting indices of the segments in all samples for a client.包含客户端所有样本中各段起始索引的列表。
        seg_len: An integer indicating the length of a segment  一个整数，表示每个段的长度
        batch_size: An integer indicating the number of batches  一个整数，表示批次的数量

    Returns:
        A tuple containing the batches of sequences of modalities A&B and labels y
        一个包含模态A和B序列以及标签y的批次的元组。
    """
    samples_A = dataset["A"]
    samples_B = dataset["B"]
    samples_y = dataset["y"]

    input_size_A = len(samples_A[0])
    input_size_B = len(samples_B[0])
    # the length of each sequence 计算每个序列的长度 （一般为，（数据的长度*1）/batch）
    seq_len = seg_len * len(seg_idxs) // batch_size
    #确保每个序列的长度不超过单个段的长度，以避免超过可用的数据范围。
    if seq_len > seg_len:
        seq_len = seg_len - 1
    # 计算所有可能的起始索引
    all_indices_start = []
    for idx in seg_idxs:
        indices_start_in_seg = list(range(idx, idx + seg_len - seq_len))
        all_indices_start.extend(indices_start_in_seg)
    # 从所有可能的起始索引中随机选择batch_size个索引
    indices_start = np.random.choice(
        all_indices_start, batch_size, replace=False)

    A_seq = np.zeros((batch_size, seq_len, input_size_A), dtype=np.float32)
    B_seq = np.zeros((batch_size, seq_len, input_size_B), dtype=np.float32)
    y_seq = np.zeros((batch_size, seq_len), dtype=np.uint8)

    for i in range(batch_size):
        idx_start = indices_start[i]
        idx_end = idx_start+seq_len
        A_seq[i, :, :] = samples_A[idx_start: idx_end, :]
        B_seq[i, :, :] = samples_B[idx_start: idx_end, :]
        y_seq[i, :] = samples_y[idx_start:idx_end]
    return (A_seq, B_seq, y_seq)


def client_idxs(data_train, config):  #就是在为每个客户端从大数据集里面分配数据（返回的是分配得到的数据起始索引），现在得到的数据索引都包含A、B所有模态的数据
    #它将训练数据划分为段，并在每个段内为每个客户端随机选择一个起始位置。结果是一个列表，其中每个元素对应一个客户端，包含表示训练数据中段的起始位置的列表。
    """Generates sample indices for each client.

    Args:
        data_train: a dictionary containing training data of modalities A&B and labels y
        config: a map of configurations of the simulation

    Returns:
    A list containing the sample indices for each client. Each item in the list is a list of numbers and each number representing the starting location of a segment in the training data.
    """
    # 提取每个模态的客户端数量和总客户端数量
    num_clients_A = int(config["FL"]["num_clients_A"])
    num_clients_B = int(config["FL"]["num_clients_B"])
    num_clients_AB = int(config["FL"]["num_clients_AB"])
    num_clients = num_clients_A+num_clients_B+num_clients_AB

    # 获取训练数据中的样本数量
    n_samples = len(data_train["A"])  # number of rows in training data
    # divide the training data into divisions
    if config["SIMULATION"]["data"] == "opp":
        n_div = N_DIV_OPP
    elif config["SIMULATION"]["data"] == "mhealth":
        n_div = N_DIV_MHEALTH
    elif config["SIMULATION"]["data"] == "ur_fall":
        n_div = N_DIV_URFALL


    # each client has (n_samples * train_ratio) data
    train_ratio = float(config["FL"]["train_ratio"])

    len_div = int(n_samples // n_div)  # the length of each division
    # Within each division, we randomly pick 1 segment. So the length of each segment is
    len_seg = get_seg_len(n_samples, config)
    starts_div = np.arange(0, n_samples-len_div, len_div)
    idxs_clients = []
    for i in range(num_clients):
        idxs_clients.append(np.array([]).astype(np.int64))
        for start in starts_div:
            idxs_in_div = np.arange(start, start + len_div - len_seg)
            idxs_clients[i] = np.append(
                idxs_clients[i], np.random.choice(idxs_in_div))
    return idxs_clients


def download_UR_fall():  #下载ur_fall数据集，但此函数不好用，因为要一个一个的创建路径（用ur_fall_test1.py来下载）
    """Downloads the UR Fall datasets from http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html"""
    url = "http://fenix.univ.rzeszow.pl/~mkepski/ds"
    # 循环遍历下载Fall数据集的文件（i从1到30），30个跌倒检测
    for i in range(1, 31):
        print(f"Downloading files {i}")
        depth_camera_0 = f"{url}/data/fall-{str(i).zfill(2)}-cam0-d.zip"
        depth_camera_1 = f"{url}/data/fall-{str(i).zfill(2)}-cam1-d.zip"
        rgb_camera_0 = f"{url}/data/fall-{str(i).zfill(2)}-cam0-rgb.zip"
        rgb_camera_1 = f"{url}/data/fall-{str(i).zfill(2)}-cam1-rgb.zip"
        sync_file = f"{url}/data/fall-{str(i).zfill(2)}-data.csv"
        acc_file = f"{url}/data/fall-{str(i).zfill(2)}-acc.csv"
        # 使用requests库下载文件并保存到本地目录
        r = requests.get(depth_camera_0)
        open(
            f"download/UR_FALL/fall/cam0-d/fall-{str(i).zfill(2)}-cam0-d.zip", "wb").write(r.content)

        r = requests.get(depth_camera_1)
        open(
            f"download/UR_FALL/fall/cam1-d/fall-{str(i).zfill(2)}-cam1-d.zip", "wb").write(r.content)
        r = requests.get(rgb_camera_0)
        open(
            f"download/UR_FALL/fall/cam0-rgb/fall-{str(i).zfill(2)}-cam0-rgb.zip", "wb").write(r.content)
        r = requests.get(rgb_camera_1)
        open(
            f"download/UR_FALL/fall/cam1-rgb/fall-{str(i).zfill(2)}-cam1-rgb.zip", "wb").write(r.content)
        r = requests.get(sync_file)
        open(
            f"download/UR_FALL/fall/sync/fall-{str(i).zfill(2)}-data.csv", "wb").write(r.content)
        r = requests.get(acc_file)
        open(
            f"download/UR_FALL/fall/acc/fall-{str(i).zfill(2)}-acc.csv", "wb").write(r.content)

    # 循环遍历下载ADL数据集的文件（i从1到40），40个日常活动
    for i in range(1, 41):
        print(f"Downloading files {i}")
        depth_camera_0 = f"{url}/data/adl-{str(i).zfill(2)}-cam0-d.zip"
        rgb_camera_0 = f"{url}/data/adl-{str(i).zfill(2)}-cam0-rgb.zip"
        sync_file = f"{url}/data/adl-{str(i).zfill(2)}-data.csv"
        acc_file = f"{url}/data/adl-{str(i).zfill(2)}-acc.csv"
        r = requests.get(depth_camera_0)
        open(
            f"download/UR_FALL/adl/cam0-d/adl-{str(i).zfill(2)}-cam0-d.zip", "wb").write(r.content)
        r = requests.get(rgb_camera_0)
        open(
            f"download/UR_FALL/adl/cam0-rgb/adl-{str(i).zfill(2)}-cam0-rgb.zip", "wb").write(r.content)
        r = requests.get(sync_file)
        open(
            f"download/UR_FALL/adl/sync/adl-{str(i).zfill(2)}-data.csv", "wb").write(r.content)
        r = requests.get(acc_file)
        open(
            f"download/UR_FALL/adl/acc/adl-{str(i).zfill(2)}-acc.csv", "wb").write(r.content)
    # 下载提取的特征文件
    print("Downloading extracted features")
    features_fall = f"{url}/data/urfall-cam0-falls.csv"
    r = requests.get(features_fall)
    open(f"download/UR_FALL/fall/urfall-features-cam0-falls.csv",
         "wb").write(r.content)

    features_adl = f"{url}/data/urfall-cam0-adls.csv"
    r = requests.get(features_adl)
    open(f"download/UR_FALL/adl/urfall-features-cam0-adls.csv", "wb").write(r.content)


if __name__ == "__main__":
    #gen_opp("data")
    #gen_mhealth(r"D:\code_home\ECE535-FederatedLearning-main\data")
    #download_UR_fall()
    #gen_ur_fall(r"D:\code_home\ECE535-FederatedLearning-main\data")
    pass
