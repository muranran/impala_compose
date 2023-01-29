import multiprocessing as mp
from multiprocessing import Process
import numpy as np
from time import time, sleep
import pyarrow as pa
import shared_numpy as snp
def child(q, ):
    data = q.get()
    
    for i in range(100):
        q.get()
        print(data)
        # sleep(0.1)
    pass





if __name__ == "__main__":
    # q = snp.Queue()
    # data = snp.ndarray((10,10,10,10),dtype=np.float32)
    
    # p=Process(target=child,args=(q,))
    # p.start()
    # q.put(data)
    # for i in range(100):
    #     q.put("yes")
    #     x = np.ones((10,10,10,20))*i
    #     data[:]= x[:][:][:][10:]
    #     sleep(0.1)
    # p.join()
    # data.close()
    # data.unlink()
    data = [np.ones((120,10)),np.ones((120,4)),np.ones((120,8))]
    data2 = data[:][:60]
    print(data2)