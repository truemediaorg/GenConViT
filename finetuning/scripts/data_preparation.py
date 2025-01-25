import os
import shutil
import random

def split():
    # File paths
    eval_ids_path = '/home/ubuntu/lin/global-ml/GenConViT/finetuning/eval_ids.txt'
    data_dir = '/home/ubuntu/lin/global-ml/GenConViT/finetuning/all_data'
    output_base = '/home/ubuntu/lin/global-ml/GenConViT/finetuning/new_data'

    # Statistics dictionary to keep track of real and fake counts
    stats = {
        'train_vid': {'real': 0, 'fake': 0},
        'test_vid': {'real': 0, 'fake': 0},
        'val_vid': {'real': 0, 'fake': 0}
    }

    # Create directories if they don't exist
    for subfolder in ['train_vid', 'test_vid', 'val_vid']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(output_base, subfolder, label), exist_ok=True)

    # Read filenames from txt file
    with open(eval_ids_path, 'r') as file:
        filenames = file.read().splitlines()

    # Shuffle the list
    random.shuffle(filenames)

    # Calculate split indices
    total_files = len(filenames)
    train_end = int(total_files * 0.6)
    test_end = train_end + int(total_files * 0.2)

    # Splitting the filenames
    train_files = filenames[:train_end]
    test_files = filenames[train_end:test_end]
    val_files = filenames[test_end:]

    # Function to copy files and count labels
    def copy_files(files, subfolder):
        for entry in files:
            filename, label = entry.split('\t')  # Adjusted to split by tab character
            label = label.lower().strip()
            if label not in ['real', 'fake']:
                print(f'Unexpected label found: {label}')
                continue

            source_path = os.path.join(data_dir, filename)
            destination_path = os.path.join(output_base, subfolder, label)

            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                stats[subfolder][label] += 1
            else:
                print(f'File not found: {filename} file path = {source_path} ')

    # Copy files to respective folders and count them
    copy_files(train_files, 'train_vid')
    copy_files(test_files, 'test_vid')
    copy_files(val_files, 'val_vid')

    total_fake = 0
    total_real = 0
    # Print out the stats
    for category, counts in stats.items():
        real_count = counts['real']
        fake_count = counts['fake']
        total_fake += fake_count
        total_real += real_count
        print(f'{category}: total = {real_count + fake_count} real = {real_count} fake = {fake_count}.')

    print(f"Total Fake = {total_fake} Real = {total_real}")


    print('Files have been copied to respective folders and categorized by label.')


def split_all():
    import os
    import shutil

    # File paths
    eval_ids_path = '/home/ubuntu/lin/global-ml/GenConViT/finetuning/eval_ids.txt'
    data_dir = '/home/ubuntu/lin/global-ml/GenConViT/finetuning/all_data'
    output_base = '/home/ubuntu/lin/global-ml/GenConViT/finetuning/unified_data'

    # Create directories for 'real' and 'fake' if they don't exist
    for label in ['real', 'fake']:
        os.makedirs(os.path.join(output_base, label), exist_ok=True)

    # Statistics dictionary to keep track of real and fake counts
    stats = {'real': 0, 'fake': 0}

    # Read filenames from the txt file
    with open(eval_ids_path, 'r') as file:
        entries = file.read().splitlines()

    # Function to copy files and count labels
    def copy_files(entries):
        for entry in entries:
            filename, label = entry.split('\t')  # Assuming tab-separated filename and label
            label = label.lower().strip()
            if label not in ['real', 'fake']:
                print(f'Unexpected label found: {label}')
                continue

            source_path = os.path.join(data_dir, filename)
            destination_path = os.path.join(output_base, label, filename)

            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                stats[label] += 1
            else:
                print(f'File not found: {filename} file path = {source_path}')

    # Copy files to respective folders and count them
    copy_files(entries)

    # Print out the stats
    print(f"Real files: {stats['real']}")
    print(f"Fake files: {stats['fake']}")

    print('Files have been copied to respective folders and categorized by label.')


def main():
    split_all()
    
if __name__ == "__main__":
    main()