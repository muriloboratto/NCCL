#NCCL

----
## what is NCCL?
see [NVIDIA](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/index.html)

> NCCL is a A sample of how to call collective operation functions on multi-GPU. A simple example of using broadcast, reduce, allGather, reduceScatter operations.

----

## Introduction

The NVIDIA create a friendly solution to use this interconnect issue by providing higher bandwidth that call NVIDIA Collective Communications Library (NCCL). This library provides routines such as all-gather, all-reduce, broadcast, reduce, reduce-scatter, that are optimized to achieve high bandwidth over PCIe and NVLINK high-speed interconnect and implements multi-GPU and multi-node collective communication primitives that are performance optimized for NVIDIA GPUs on NVLINK technology to interconnects.  NCCL is a library of multi-GPU collective communication primitives that are topology-aware and can be easily integrated into your application. Initially developed as an open-source research project, NCCL is designed to be light-weight, depending only on the usual C++ and CUDA libraries.

----

## Collective Operations

At present, the library implements the following collectives operations:

* all-reduce
* all-gather
* reduce-scatter
* reduce
* broadcast

----

## Requirements

NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. For PCIe based platforms, best performance is achieved when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.
