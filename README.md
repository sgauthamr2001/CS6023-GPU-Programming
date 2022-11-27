# CS6023-GPU-Programming

<p align = "justify"> The repository contains assignments done in CUDA as a part of the course GPU Programming, Jan-May 2022. The problem statement of each assignment has been attached in the sub-folders. A brief description about them has been presented. Instructions for custom self-checking:</p>

> 1. Add the .cu file to the evaluation-script repository. 
> 2. Add the custom input as .txt to evaluation-script/testcases/input, refer to problem statement for input format.
> 3. Add the corresponding output to evaluation-script/testcases/output. 
> 4. Execute evaluate.sh. 

Note: Testing setup was taken from the [course materials](http://www.cse.iitm.ac.in/~rupesh/teaching/gpu/jan22/).

#### Assignment - 1
> Problem Statement: Involves the computation of $(A + B^{T})(B^{T} - A)$ using parallel computation. 
<p align = "justify">Profiling only the computation, the speed-up obtained by using a GPU over the CPU baseline for large test-case was $87.72$ times, while for a small test cases, both the cases almost take similar amount of time . This is because on the CPU, the code is $O(mn)$, hence large matrices shall compensate beyond the reduction is clock, while executing on a GPU.  </p>

#### Assignment - 2
> Problem Statement: Parallelise the computation of $X$ = $(A + B^{T})CD^{T}$. Parallise the application considering memory coalescing, shared memory, degree of divergence. 
<p align = "justify"> Parallelisation of GEMM, along with the usage of shared memory and tiling to improve coalescing. In this assignment all the matrices are processed as a $32$ x $32$ tile, to get better coalescing, and bank conflicts are kept minimised. The computation of $X$ is broken down into three computations, $A + B^{T}$ is computed first followed by successive multiplications on the right. Each of these stages has been optimised using shared memory, coalesced acesses and reduced bank conflicts to gain an improvement of $0.257$ ms, $14.592$ ms and $86.554$ ms in each of the stages of computation in comparision to the base case of naive parallelisation on GPU. </p>

#### Assignment - 3
> Problem Statement:  Given a set of M cores and N tasks, as well as the time of execution $T(i)$ and priorities, the turn-around time of each of the tasks is to be computed. 
<p align = "justify"> For the case of task scheduling multiple shared variables are needed given multiple tasks could be scheduled to the same core. This requires synchronisation across threads for functional correctness. To improve the performance, in-order to find a free core, the search is performed in a reductive fashion. </p>

#### Course_Project | Assignment - 4
> Problem Statement:  Given a set of N trains and M classes, as well source, destination, capacity of each of the train, the task is to process B batches of requests, each containing R requests, where all of them are passed as input. 

<p align = "justify"> For gaining on terms of performance, requests to same class in a train are sequentialised while other requests are processed parallely. For sequentialising requests belonging to same class in a train, a shared array link is maintained. For say, requests 0, 1, 9 fall in this category, then link[0] holds 1, link[1] holds 9. This enables the sequentialsing of requests. While this is the case for same train and class, the others requests are processed parallely. The link array is maintained by efficient usage of the shared memory.</p>

