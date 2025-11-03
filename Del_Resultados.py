import os
import shutil

datasets = ['amazon-books', 'kuaisim']
#'RetailRocket-Transactions', 'DeliciousBookmarks', 'MovieLens', 'BestBuy',
#  'Taobao', 'Events', 'CiaoDVD', 'NetflixPrize'

recommenders = ['Item2Vec_itemSim', 'TimeI2V_Disc', 'TimeI2V_Cont']
# 'ALS', 'BPR'
# 'ALS_itemSim', 'BPR_itemSim',
# 'ALS_itemSim_temporal', 'BPR_itemSim_temporal', 
# 'Item2Vec_itemSim', 'TimeI2V_Disc', 'TimeI2V_Disc_Aug', 'TimeI2V_Cont'


def remove_folders_by_name(results_folder, names):
    for root, dirs, files in os.walk(results_folder):
        for dir_name in dirs:
            for name in names:
                if name in dir_name:
                    dir_path = os.path.join(root, dir_name)
                    shutil.rmtree(dir_path)
                    print(f"Removed folder: {dir_path}")

results_folder = "results"

for subdir1 in os.listdir(results_folder):
    subdir1_path = os.path.join(results_folder, subdir1)
    if os.path.isdir(subdir1_path):
        for subdir2 in os.listdir(subdir1_path):
            subdir2_path = os.path.join(subdir1_path, subdir2)
            if os.path.isdir(subdir2_path):
                for dataset_name in datasets:
                    dataset_folder = os.path.join(subdir2_path, dataset_name)
                    if os.path.exists(dataset_folder):
                        if 'all' in recommenders:
                            shutil.rmtree(dataset_folder)
                            print(f"Removed entire dataset folder: {dataset_folder}")
                        else:
                            remove_folders_by_name(dataset_folder, recommenders)
                    else:
                        print(f"Dataset folder '{dataset_name}' does not exist in {subdir2}.")