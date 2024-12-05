import socket
import threading
import time
import torch
import ctypes
import sys
sys.path.insert(0, '/root/python-nccl')
from src.pynccl_wrapper import *

class NCCLWorker:
    def __init__(self, controller_host='localhost', controller_port=5000):
        self.nccl_lib = NCCLLibrary()
        self.nccl_comm = None
        self.rank = -1
        self.size = 0
        self.reinit_flag = False
        self.unique_id = None
        self.controller_host = controller_host
        self.controller_port = controller_port
        self.sock = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.controller_host, self.controller_port))
        print("Connected to controller.")

        threading.Thread(target=self.heartbeat_sender, daemon=True).start()
        threading.Thread(target=self.rank_listener, daemon=True).start()
        threading.Thread(target=self.main_worker_logic, daemon=True).start()

    def heartbeat_sender(self):
        try:
            while True:
                self.sock.sendall("heartbeat".encode('utf-8'))
                time.sleep(5)
        except (ConnectionResetError, BrokenPipeError):
            print("Lost connection to controller.")
            self.reinit_flag = True

    def rank_listener(self):
        try:
            while True:
                data = self.sock.recv(1024).decode('utf-8')
                if data.startswith("RANK_UPDATE"):
                    parts = data.split()
                    self.rank = int(parts[1])
                    self.size = int(parts[2])
                    print(f"Rank updated: {self.rank}/{self.size}")

                    if self.rank == 0:
                        self.unique_id = self.nccl_lib.ncclGetUniqueId()
                        unique_id_buffer = ctypes.string_at(ctypes.byref(self.unique_id), ctypes.sizeof(self.unique_id))
                        self.sock.sendall(unique_id_buffer)
                    else:
                        unique_id_buffer = self.sock.recv(128)
                        self.unique_id = ncclUniqueId()
                        ctypes.memmove(ctypes.byref(self.unique_id), unique_id_buffer, ctypes.sizeof(self.unique_id))
                    self.reinit_flag = True
        except (ConnectionResetError, BrokenPipeError):
            print("Lost connection to controller.")
            self.reinit_flag = True

    def WaitForSyncForInit(self):
        while True:
            state = self.nccl_lib.ncclCommGetAsyncError(self.nccl_comm).value
            if state != 7:
                break

    def initialize_nccl(self):
        config = NCCL_CONFIG_INITIALIZER()
        config.blocking = 0
        torch.cuda.set_device(0)
        print(f'my rank:{self.rank}, world size:{self.size}')
        self.nccl_comm = self.nccl_lib.ncclCommInitRankConfig(self.size, self.unique_id, self.rank, config)
        self.WaitForSyncForInit()
        print(f"Rank {self.rank}: NCCL initialization finished")
        self.reinit_flag = False

    def ReInitNccl(self):
        if self.reinit_flag:
            print("Reinitializing NCCL...")
            self.nccl_lib.ncclCommAbort(self.nccl_comm)
            self.initialize_nccl()
            self.reinit_flag = False

    def WaitForSync(self):
        while True:
            state = self.nccl_lib.ncclCommGetAsyncError(self.nccl_comm).value
            if state != 7:
                break
            self.ReInitNccl()

    def main_worker_logic(self):
        while self.rank == -1 or self.unique_id is None:
            time.sleep(3)

        self.initialize_nccl()

        kv_cache_shape = (1, 2, 3)
        device = torch.device(f'cuda:0')
        dtype = torch.float32
        kv_cache = torch.full(kv_cache_shape, fill_value=42, dtype=dtype, device=device)
        recv_tensor = torch.zeros_like(kv_cache)

        sendbuff = kv_cache.data_ptr()
        recvbuff = recv_tensor.data_ptr()
        count = kv_cache.numel()
        datatype = ncclDataTypeEnum.ncclFloat32

        time.sleep(10)
        while True:
            time.sleep(1)
            if self.rank == 0:
                print("send")
                for i in range(1, self.size):
                    self.ReInitNccl()
                    self.nccl_lib.ncclSend(sendbuff, count, datatype, i, self.nccl_comm, ctypes.c_void_p(0))
                    self.WaitForSync()
            else:
                print("recv")
                self.ReInitNccl()
                self.nccl_lib.ncclRecv(recvbuff, count, datatype, 0, self.nccl_comm, ctypes.c_void_p(0))
                self.WaitForSync()
            print(f"Rank {self.rank}: Operation complete, result {recv_tensor}")

if __name__ == "__main__":
    worker = NCCLWorker()
    worker.start()
