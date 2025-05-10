import os
import shutil


def remove_empty_rgb_folders(base_path):
    """
    Traverse through each scene folder under the base path and remove the scene folder if the 'rgb' folder is empty.

    Parameters:
    - base_path (str): The base directory path containing scene folders.
    """
    # Get all scene folders in the base path
    for scene_name in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_name)

        # Ensure we are dealing with directories
        if os.path.isdir(scene_path):
            rgb_path = os.path.join(scene_path, 'rgb')

            # Check if 'rgb' folder exists and if it is empty
            if os.path.exists(rgb_path) and os.path.isdir(rgb_path):
                if not os.listdir(rgb_path):  # Check if the 'rgb' folder is empty
                    print(f"Removing empty scene folder: {scene_path}")
                    shutil.rmtree(scene_path)  # Remove the scene folder


# Usage
base_path = '/path/to/NeuGrasp/data/NeuGraspData/data/train_raw_mix_real'  # Update this to your actual base path
remove_empty_rgb_folders(base_path)
