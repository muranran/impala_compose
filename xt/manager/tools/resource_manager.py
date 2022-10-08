import logging

import psutil
from multiprocessing import Queue
DATA_STORE_THRESHOLD = 100


def resource_monitor(data_queue=None, auto_clear=False):
    avail_mem = psutil.virtual_memory().available // (8 * 1000 * 1000)
    total_mem = psutil.virtual_memory().total // (8 * 1000 * 1000)

    if avail_mem / total_mem < 0.2:
        raise ResourceWarning("Available mem of server is less than 20% !!!")

    if data_queue is not None:
        if isinstance(data_queue, Queue):
            case_num = data_queue.qsize()
            logging.info("Untreated Data Case Number: {}\t{:.2f}% of All".format(case_num, case_num))

            if auto_clear:
                if avail_mem / total_mem < 0.3:
                    logging.info("Unused Mem {:.2f}%, Clear Buffer".format(avail_mem / total_mem * 100))
                    while not data_queue.empty():
                        data_queue.get(False)



