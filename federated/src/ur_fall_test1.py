import os
import requests

def download_UR_fall(data_path):
    """从http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html下载UR Fall数据集

    参数:
        data_path (str): 数据集下载的路径.
    """
    # 数据集的URL
    url = "http://fenix.univ.rzeszow.pl/~mkepski/ds"

    # 循环下载fall数据集
    for i in range(1, 31):
        print(f"正在下载文件 {i}")

        # fall文件夹
        fall_folder = os.path.join(data_path, "ur_fall/fall")
        os.makedirs(fall_folder, exist_ok=True)

        # 为每种类型的数据集创建子文件夹
        subfolders = ["cam0-d", "cam1-d", "cam0-rgb", "cam1-rgb", "sync", "acc"]
        for subfolder in subfolders:
            os.makedirs(os.path.join(fall_folder, subfolder), exist_ok=True)

        # 下载深度相机0数据
        depth_camera_0 = f"{url}/data/fall-{str(i).zfill(2)}-cam0-d.zip"
        r = requests.get(depth_camera_0)
        open(os.path.join(fall_folder, "cam0-d", f"fall-{str(i).zfill(2)}-cam0-d.zip"), "wb").write(r.content)

        # 为其他数据集重复此过程
        depth_camera_1 = f"{url}/data/fall-{str(i).zfill(2)}-cam1-d.zip"
        r = requests.get(depth_camera_1)
        open(os.path.join(fall_folder, "cam1-d", f"fall-{str(i).zfill(2)}-cam1-d.zip"), "wb").write(r.content)

        rgb_camera_0 = f"{url}/data/fall-{str(i).zfill(2)}-cam0-rgb.zip"
        r = requests.get(rgb_camera_0)
        open(os.path.join(fall_folder, "cam0-rgb", f"fall-{str(i).zfill(2)}-cam0-rgb.zip"), "wb").write(r.content)

        rgb_camera_1 = f"{url}/data/fall-{str(i).zfill(2)}-cam1-rgb.zip"
        r = requests.get(rgb_camera_1)
        open(os.path.join(fall_folder, "cam1-rgb", f"fall-{str(i).zfill(2)}-cam1-rgb.zip"), "wb").write(r.content)

        sync_file = f"{url}/data/fall-{str(i).zfill(2)}-data.csv"
        r = requests.get(sync_file)
        open(os.path.join(fall_folder, "sync", f"fall-{str(i).zfill(2)}-data.csv"), "wb").write(r.content)

        acc_file = f"{url}/data/fall-{str(i).zfill(2)}-acc.csv"
        r = requests.get(acc_file)
        open(os.path.join(fall_folder, "acc", f"fall-{str(i).zfill(2)}-acc.csv"), "wb").write(r.content)

    # 循环下载adl数据集
    for i in range(1, 41):
        print(f"正在下载文件 {i}")

        # adl文件夹
        adl_folder = os.path.join(data_path, "ur_fall/adl")
        os.makedirs(adl_folder, exist_ok=True)

        # 为每种类型的数据集创建子文件夹
        subfolders = ["cam0-d", "cam0-rgb", "sync", "acc"]
        for subfolder in subfolders:
            os.makedirs(os.path.join(adl_folder, subfolder), exist_ok=True)

        # 下载深度相机0数据
        depth_camera_0 = f"{url}/data/adl-{str(i).zfill(2)}-cam0-d.zip"
        r = requests.get(depth_camera_0)
        open(os.path.join(adl_folder, "cam0-d", f"adl-{str(i).zfill(2)}-cam0-d.zip"), "wb").write(r.content)

        # 为其他数据集重复此过程
        rgb_camera_0 = f"{url}/data/adl-{str(i).zfill(2)}-cam0-rgb.zip"
        r = requests.get(rgb_camera_0)
        open(os.path.join(adl_folder, "cam0-rgb", f"adl-{str(i).zfill(2)}-cam0-rgb.zip"), "wb").write(r.content)

        sync_file = f"{url}/data/adl-{str(i).zfill(2)}-data.csv"
        r = requests.get(sync_file)
        open(os.path.join(adl_folder, "sync", f"adl-{str(i).zfill(2)}-data.csv"), "wb").write(r.content)

        acc_file = f"{url}/data/adl-{str(i).zfill(2)}-acc.csv"
        r = requests.get(acc_file)
        open(os.path.join(adl_folder, "acc", f"adl-{str(i).zfill(2)}-acc.csv"), "wb").write(r.content)

    # 下载fall数据集的提取特征
    print("正在下载提取的特征")
    features_fall = f"{url}/data/urfall-cam0-falls.csv"
    r = requests.get(features_fall)
    open(os.path.join(fall_folder, "urfall-features-cam0-falls.csv"), "wb").write(r.content)

    # 下载adl数据集的提取特征
    features_adl = f"{url}/data/urfall-cam0-adls.csv"
    r = requests.get(features_adl)
    open(os.path.join(adl_folder, "urfall-features-cam0-adls.csv"), "wb").write(r.content)

# 使用
download_UR_fall(r"D:/code_home/ECE535-FederatedLearning-main/data/")

