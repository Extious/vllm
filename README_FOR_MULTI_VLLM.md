# About
Forked from vllm.
Here is the branch multi-vllm, all is modified from v0.6.4.
We implement nccl communication between multiple instances of vllm at this branch.
# Introduction
To realise the nccl communication, we use the project [python-nccl](https://github.com/freshduer/python-nccl) and you directory structure can be like this:
```bash
[username@host multi-vllm]$ tree -L 2
.
.
|-- python-nccl
|   |-- build
|   |-- examples
|   |-- ln.sh
|   |-- mpi_examples
|   |-- nccl_wrapper.cpython-312-x86_64-linux-gnu.so
|   |-- nccl_wrapper.egg-info
|   |-- README.md
|   |-- setup.py
|   `-- src
`-- vllm
    |-- benchmarks
    |-- cmake
    |-- CMakeLists.txt
    |-- CODE_OF_CONDUCT.md
    |-- collect_env.py
    |-- CONTRIBUTING.md
    |-- csrc
    |-- DCO
    |-- Dockerfile
    |-- Dockerfile.cpu
    |-- Dockerfile.hpu
    |-- Dockerfile.neuron
    |-- Dockerfile.openvino
    |-- Dockerfile.ppc64le
    |-- Dockerfile.rocm
    |-- Dockerfile.tpu
    |-- Dockerfile.xpu
    |-- docs
    |-- examples
    |-- find_cuda_init.py
    |-- format.sh
    |-- LICENSE
    |-- MANIFEST.in
    |-- pyproject.toml
    |-- python_only_dev.py
    |-- README_FOR_MULTI_VLLM.md
    |-- README.md
    |-- requirements-build.txt
    |-- requirements-common.txt
    |-- requirements-cpu.txt
    |-- requirements-cuda.txt
    |-- requirements-dev.txt
    |-- requirements-hpu.txt
    |-- requirements-lint.txt
    |-- requirements-neuron.txt
    |-- requirements-openvino.txt
    |-- requirements-rocm.txt
    |-- requirements-test.in
    |-- requirements-test.txt
    |-- requirements-tpu.txt
    |-- requirements-xpu.txt
    |-- SECURITY.md
    |-- setup.py
    |-- tests
    |-- tools
    |-- use_existing_torch.py
    |-- vllm
    `-- vllm.egg-info

```
You can ``mkdir multi-vllm`` on your machine and ``cd multi-vllm``. In this directory you can clone [python-nccl](https://github.com/freshduer/python-nccl) and this project.
You should build [python-nccl](https://github.com/freshduer/python-nccl) on your machine at first and then you can import the nccl library in the vllm.
# Usage
There can be  a controller and multi vllm engines. You can create a conda env for this project.
To run the demo, you can enter the following command in a terminal to run the controller:
```bash
[username@host multi-vllm]$ python vllm/vllm/python_nccl/controller.py
```
The controller is Responsible for UniqueID distribution and initialization.
Then you can run two vllm instances in two another terminals.
```bash
CUDA_VISIBLE_DEVICES=0  python vllm/vllm/entrypoints/openai/api_server.py --model meta-llama/Llama-3.1-8B-Instruct
```
```bash
CUDA_VISIBLE_DEVICES=1 python vllm/vllm/entrypoints/openai/api_server.py --model meta-llama/Llama-3.1-8B-Instruct --port 8001
```
You can use ``git diff`` to get the difference between this branch and v0.6.4.
# Write at the end
This is just a demo that proves it is feasible to use nccl communication between multiple vllm instances. While the two vllm instances is running, there are always tensors transferred between them. When and where to transfer the KV caches, checkpoint etc. to realise reusing them on multi vllm instances is the future. 