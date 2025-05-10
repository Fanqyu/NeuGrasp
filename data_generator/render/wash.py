import sys
sys.path.append('/path/to/NeuGrasp')

import os
from tqdm import tqdm
from colorama import *


if __name__ == '__main__':
    root = '/path/to/NeuGraspData/data/giga_hemisphere_train_0827/packed_full/6_M_rand'
    count = 0
    for split_name in tqdm(os.listdir(root)):
        split_path = os.path.join(root, split_name)
        for scene_name in tqdm(os.listdir(split_path)):
            scene_path = os.path.join(split_path, scene_name)
            if 'depth' not in os.listdir(scene_path):
                print(scene_name, ': no depth')
                count += 1
                print(scene_name, ' deleting')
                os.system('sudo rm -r ' + scene_path)
                print(Fore.GREEN + 'Successfully' + Fore.RESET)
            else:
                depth_path = os.path.join(scene_path, 'depth')
                n = len(os.listdir(depth_path))
                if n < 24:
                    print(scene_name, ': num_depths less than 24')
                    count += 1
                    print(scene_name, ' deleting')
                    os.system('sudo rm -r ' + scene_path)
                    print(Fore.GREEN + 'Successfully' + Fore.RESET)
    print('all' + str(count) + 'are deleted.')


