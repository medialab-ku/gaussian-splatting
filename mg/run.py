import os
import cv2
from multiprocessing import Process, Queue
from tum_dataset import TumDataset
from tracking import Tracker
import torch
import torch.multiprocessing as mp

def PlayTumDataset(q):
    dataset = TumDataset()
    begin_index = 1000
    cnt = 1000
    for index in range (cnt):
        rgb, gray, d = dataset.ReturnData(index + begin_index)
        # cv2.imshow("test", gray)
        # cv2.waitKey(10)
        q.put([True, rgb, gray, d])
    q.put([False])

def TrackingTest(q):
    tracker = Tracker()
    while True:
        if not q.empty():
            instance = q.get()
            if not instance[0]:
                return
            tracker.Track(instance[1], instance[2], instance[3])



if __name__ == '__main__':
    mp_manager = mp.Manager()
    img_pair_q = mp.Queue()
    process_play_data = mp.Process(target=PlayTumDataset, args=(img_pair_q,))
    process_tracking = mp.Process(target=TrackingTest, args=(img_pair_q,))
    # process_mapping = Process(target=my_consumer, args=(q,))
    # process_gaussian = Process(target=my_consumer, args=(q,))
    process_play_data.start()
    process_tracking.start()
    # process_mapping.start()
    # process_gaussian.start()
    #
    # q.close()
    # q.join_thread()
    #
    process_play_data.join()
    process_tracking.join()
    # process_mapping.join()
    # process_gaussian.join()
