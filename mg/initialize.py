import math
from replica_dataset import ReplicaDataset
from tum_dataset import TumDataset

# TODO: make initialize

def GetDataset(dataset_type):
    dataset_dict = {
        "TUM": TumDataset(),
        "REPLICA": ReplicaDataset()
        # "SCANNET" : ScannetDataset()
    }
    dataset = dataset_dict.get(dataset_type)
    return dataset

if __name__ == '__main__':
    # EXAMPLE: TUM, REPLICA, SCANNET
    dataset_type = "REPLICA"
    dataset = GetDataset(dataset_type)
    dataset.InitializeDataset()
    print(dataset.get_data_len())
