import glob
import random
import time

import multiprocessing

import requests

HOST = '127.0.0.1'
PORT = 8000

def task():
    num_request = 1000
    file_lst = glob.glob('data/MNIST/raw/samples/*.jpeg')
    for i in range(num_request):
        file_name = random.choice(file_lst)
        files = {'file': open(file_name, 'rb')}
        response = requests.post(url="http://{}:{}/inference/inference".format(HOST, PORT), files=files)
        print(file_name, response.text)

t = time.time()
processes = []
for i in range(10):
    p = multiprocessing.Process(target = task)
    p.start()
    processes.append(p)
for p in processes:
    p.join()

print('total time: ', time.time()-t)