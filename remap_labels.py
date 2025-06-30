import numpy as np

data_dir = input("Input data folder:")
file_ids = input("Input used subset IDs, separated by commas:").split(',')

remap = {}
label_cnt = 0
for file_id in file_ids:
    train_file_path = data_dir + "/train/" + file_id + ".npz"
    test_file_path = data_dir + "/test/" + file_id + ".npz"
    train_data = np.load(train_file_path, allow_pickle=True)['data'].tolist()
    test_data = np.load(test_file_path, allow_pickle=True)['data'].tolist()
    for original_label in train_data['y'] + test_data['y']:
        if original_label not in remap.keys():
            remap[original_label] = label_cnt
            label_cnt += 1

print("Label re-mapping:", remap)
print("Total labels:", label_cnt)

for file_id in file_ids:
    train_file_path = data_dir + "/train/" + file_id + ".npz"
    test_file_path = data_dir + "/test/" + file_id + ".npz"
    train_data = np.load(train_file_path, allow_pickle=True)['data'].tolist()
    test_data = np.load(test_file_path, allow_pickle=True)['data'].tolist()

    new_labels = []
    for label in train_data['y']:
        new_labels.append(remap[label])
    train_data['y'] = new_labels
    train_file_path_new = data_dir + "_trim" + "/train/" + file_id + ".npz"
    with open(train_file_path_new, 'wb') as f:
        np.savez_compressed(f, data=train_data)
        
    new_labels = []
    for label in test_data['y']:
        new_labels.append(remap[label])
    test_data['y'] = new_labels
    test_file_path_new = data_dir + "_trim" + "/test/" + file_id + ".npz"
    with open(test_file_path_new, 'wb') as f:
        np.savez_compressed(f, data=test_data)