import os
import cv2
from multiprocessing import Process, Queue
from tum_dataset import TumDataset
from tracking import Tracker
from mapping import Mapper
from gaussian_mapping import GaussianMapper
import torch
import torch.multiprocessing as mp

def PlayTumDataset(q):
    dataset = TumDataset()
    begin_index = 1
    cnt = 1000
    for index in range (cnt):
        rgb, gray, d = dataset.ReturnData(index + begin_index)
        # cv2.imshow("test", gray)
        # cv2.waitKey(10)
        q.put([True, rgb, gray, d])
    # q.put([False])

def TrackingTest(img_pair_q, tracking_result_q,):
    tracker = Tracker()
    # tracker.TMPReadOneline()
    while True:
        if not img_pair_q.empty():
            instance = img_pair_q.get()
            if not instance[0]:
                # Abort
                print("Tracking Abort")
                tracking_result_q.put([False])
                return
            tracking_result = tracker.Track(instance)
            if(tracking_result[0]):
                tracking_result_q.put(tracking_result)

def MappingTest(tracking_result_q, mapping_result_q):
    mapper = Mapper()
    while True:
        if not tracking_result_q.empty():
            instance = tracking_result_q.get()
            if not instance[0]:
                print("Mapping Abort")
                # mapper.TMPBuildPointCloudAfterDone()
                return
            mapping_result = mapper.Map(instance)
            if (mapping_result[0]):
                mapping_result_q.put(mapping_result)


def GaussianMappingTest(mapping_result_q):
    gaussian_mapper = GaussianMapper()
    while True:
        if not mapping_result_q.empty():
            instance = mapping_result_q.get()
            if not instance[0]:
                print("Gaussian Mapping Abort")
                return
            gaussian_mapper.GaussianMap(instance)
        else:
            gaussian_mapper.OptimizeGaussian()
        gaussian_mapper.Visualize([])



if __name__ == '__main__':
    mp_manager = mp.Manager()
    img_pair_q = mp.Queue()
    tracking_result_q = mp.Queue()
    mapping_result_q = mp.Queue()

    process_play_data = mp.Process(target=PlayTumDataset, args=(img_pair_q,))
    process_tracking = mp.Process(target=TrackingTest, args=(img_pair_q, tracking_result_q,))
    process_mapping = mp.Process(target=MappingTest, args=(tracking_result_q, mapping_result_q,))
    process_gaussian_mapping = mp.Process(target=GaussianMappingTest, args=(mapping_result_q,))

    process_play_data.start()
    process_tracking.start()
    process_mapping.start()
    process_gaussian_mapping.start()

    process_play_data.join()
    process_tracking.join()
    process_mapping.join()
    process_gaussian_mapping.join()

