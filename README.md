# Sample Codes using NCCL on Multi-GPU

It is necessary to carry out communication operations involving multiple computational resources in most parallel applications. These communication operations can be implemented through point-to-point operations. However, this approach is not very efficient for the programmer. Parallel and distributed solutions based on collective operations have long been chosen for these applications. The MPI standard has a set of very efficient routines that perform collective operations, making better use of the computing capacity of available computational resources. Also, with the advent of new computational resources, similar routines appear for multi-GPU systems. This repository will cover the handling of NCCL routines for multi-GPU environments, constantly comparing them with the MPI standard, showing the differences and similarities between the two computational execution environments.

----
## What is NCCL?
see [NVIDIA](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/index.html)

> NCCL (NVIDIA Collective Communication Library) is a sample of how to call collective operation functions on multi-GPU. A simple example of using broadcast, reduce, allGather, reduceScatter operations.

----

## NCCL Solution

NVIDIA creates a friendly solution to this interconnect issue by providing higher bandwidth that calls NVIDIA Collective Communications Library (NCCL). This library provides routines such as all-gather, all-reduce, broadcast, reduce, reduce-scatter, optimized to achieve high bandwidth over PCIe and NVLINK high-speed interconnect and implements multi-GPU and multi-node collective communication primitives that are performance-optimized for NVIDIA GPUs on NVLINK technology to interconnects.  NCCL is a library of multi-GPU collective communication primitives that are topology-aware and easily integrated into your application. Initially developed as an open-source research project, NCCL is lightweight, depending only on the usual C++ and CUDA libraries.

----

## Collective Operations

At present, the library implements the following collectives operations:

* broadcast
* gatter
* send-recv
* reduce
* reduce-scatter

----

## Requirements

NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. For PCIe based platforms, best performance is achieved when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.
