import configparser
import argparse
import logging
import os
import warnings
import torch
#from mpi4py import MPI
from fl import FL


# script_path = os.path.abspath(__file__)
# script_dir = os.path.dirname(script_path)
# os.chdir(script_dir)
# print("cwd: ", os.getcwd())
# print("script_dir: ", script_dir)



# config_list.remove("A0_B0_AB30_label_AB_test_B")
# config_list.remove("A0_B0_AB30_label_AB_test_A")


    #print("# of configs in dir: ", len(config_list))
    #config_list.remove("A0_B0_AB30_label_A_test_B")
    # config_list.remove("A0_B0_AB30_label_AB_test_A")



def read_config(path, file):
    config = configparser.ConfigParser()
    config.read(path + file)
    return config


if __name__ == "__main__":
    # add all files from path to list
    # dataset_path = 'D:/code_home/ECE535-FederatedLearning-main/config/ur_fall/' #需要改这个路径
    # dataset_type = ['split_ae/', 'ablation/']
    # paths = ['acce_depth/', 'acce_rgb/', 'rgb_depth/']
    # config_list = []
    # count=0
    # print('ok')
    # for y in dataset_type:
    #     print(y)
    #     for x in paths:
    #         print(x)
    #         for file in os.listdir(dataset_path+y+x):
    #             count+=1
    #             print("count", count)
    #             print("config: ", dataset_path+y+x+file)
    #             config = read_config(dataset_path+y+x, file)
    #             fl = FL(config)
    #             fl.start()

    #
     # 修改后的代码只处理单个数据文件
    file = "A0_B30_AB0_label_B_test_B"  # 设置要处理的文件名
    # 构造文件的完整路径
    file_path = 'D:/code_home/federated/config/opp/dccae/acce_gyro/'
    # # 读取配置文件
    config = read_config(file_path, file)
    # # 创建FL对象并启动
    fl = FL(config)
    fl.start()



    
               
   
   
#    if file == 'A30_B30_AB0_label_B_test_A' and y == 'ablation/' and x == 'acce_gyro/':
#                 #     continue
     # config_list.append(file)
                # print(file)
            
    # print("# of configs in dir: ", len(config_list))
    
    # for file in config_list:
    #     print("\n\nrunning config file", file)
    #     config = read_config(path, file)
    #     fl = FL(config)
    #     fl.start()









# for file in config_list:
#     config = read_config(file)
#     print(file)
#     fl = FL(config)
#     fl.start()


# def read_config():
#     config = configparser.ConfigParser()
#     config.read('/Users/zach/Desktop/School/Fall2023/ECE535/ECE535-FederatedLearning/config/opp/dccae/A0_B0_AB30_label_AB_test_A')
#     return config

# config = read_config()
# fl = FL(config)
# fl.start()

# # For MPI experiments
# COMM = MPI.COMM_WORLD
# RANK = COMM.Get_rank()


# def main():
#     is_mpi = COMM.Get_size() != 1
#     config = read_config()
#     fl = FL(config, is_mpi, RANK)
#     fl.start()


# def read_config():
#     arg_parser = argparse.ArgumentParser()
#     arg_parser.add_argument("--config", type=str,
#                             help="name of the config file of simulation")
#     args = arg_parser.parse_args()
#     config = configparser.ConfigParser()
#     config.read(args.config)
#     return config


# if __name__ == "__main__":
#     main()
