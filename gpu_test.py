import torch
import time
import numpy as np
import threading
import multiprocessing as mp

def gpu_stress(duration_sec=60, tensor_size=4096, gpu_ids=None):
    if gpu_ids is None:
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices = [f'cuda:{i}' for i in gpu_ids]

    print(f"GPU Stress on devices: {devices}")

    start_time = time.time()
    gpu_tensors = []

    while time.time() - start_time < duration_sec:
        for device in devices:
            a = torch.randn((tensor_size, tensor_size), device=device)
            b = torch.randn((tensor_size, tensor_size), device=device)
            c = torch.matmul(a, b)
            gpu_tensors.append(c)

        if len(gpu_tensors) > 10:
            gpu_tensors = gpu_tensors[-5:]

def cpu_stress_worker(duration_sec=60, tensor_size=4096):
    start_time = time.time()
    cpu_arrays = []

    while time.time() - start_time < duration_sec:
        a_cpu = np.random.randn(tensor_size, tensor_size).astype(np.float32)
        b_cpu = np.random.randn(tensor_size, tensor_size).astype(np.float32)
        c_cpu = np.dot(a_cpu, b_cpu)
        cpu_arrays.append(c_cpu)

        if len(cpu_arrays) > 10:
            cpu_arrays = cpu_arrays[-5:]

def stress_system(duration_sec=60, tensor_size=4096, gpu_ids=None, cpu_processes=None):
    if cpu_processes is None:
        cpu_processes = mp.cpu_count()  # 기본은 전체 CPU 사용

    print(f"Launching {cpu_processes} CPU processes...")

    # GPU 스레드 시작
    gpu_thread = threading.Thread(target=gpu_stress, args=(duration_sec, tensor_size, gpu_ids))
    gpu_thread.start()

    # CPU 프로세스 시작
    ctx = mp.get_context('spawn')
    cpu_pool = []
    for _ in range(cpu_processes):
        p = ctx.Process(target=cpu_stress_worker, args=(duration_sec, tensor_size))
        p.start()
        cpu_pool.append(p)

    gpu_thread.join()

    for p in cpu_pool:
        p.join()

    print("All Done.")

if __name__ == "__main__":
    # 예: GPU 0과 2만 사용, CPU는 4개 프로세스
    stress_system(duration_sec=500000, tensor_size=48000, gpu_ids=[0,1,2,3,4,5,6,7], cpu_processes=16)
