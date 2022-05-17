/**
 * Title: CS6023, GPU Programming, Jan-May 2022, Assignment-1
 * Author: Sai Gautham Ravipati (EE19B053) 
 * Description: Computation of a matrix C = (A + B.T)(.)(B.T - A).

 * Modified elements: 
    - per_row_column_kernel: Each thread processes rows(columns) of A(B).   
    - per_column_row_kernel: Each thread processes columns(rows) of A(B).
    - per_element_kernel: All elements of C are computed in parallel. 
    - on_cpu: Sequential processing on CPU.
    - Functions to check execution time. 

 * Note: 
    Execution time related code is commented with "-->", uncomment 
    to print the execution time. Ananlyis of exectuion times is at 
    the end of the code. 
 **/

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

// complete the following kernel...

/**
 * Each thread processes rows(columns) of A(B). 
 **/
__global__ void per_row_column_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    long int i = blockIdx.x*blockDim.x + threadIdx.x;    // Index coressponding to row(column) of A(B)
    // Condition check as number of threads may be more than needed
    if(i < m){
        for(long int j = 0; j < n; j++){
            C[i*n+j] = (A[i*n + j] + B[j*m + i])*(B[j*m + i] - A[i*n + j]);
        }
    }
}

// complete the following kernel...

/**
 * Each thread processes columns(rows) of A(B).
 **/
__global__ void per_column_row_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    long int j = (blockIdx.x*blockDim.x + threadIdx.x)*blockDim.y + threadIdx.y;    // Index coressponding to column(row) of B(A)
    // Condition check as number of threads may be more than needed
    if(j < n){
        for(long int i = 0; i < m; i++){
            C[i*n+j] = (A[i*n + j] + B[j*m + i])*(B[j*m + i] - A[i*n + j]);
        }
    }
}

// complete the following kernel...

/**
 * All elements of C are computed in parallel. 
 **/
__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    long int i = blockIdx.y*blockDim.x + threadIdx.x;    // Index coressponding to row(column) of A(B) 
    long int j = blockIdx.x*blockDim.y + threadIdx.y;    // Index coressponding to column(row) of B(A)
    // Condition check as number of threads may be more than needed
    if(i < m && j < n){
        C[i*n+j] = (A[i*n + j] + B[j*m + i])*(B[j*m + i] - A[i*n + j]);
    }
}

/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d", stat);
  return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
	printf("%s%3f seconds\n", str, endtime - starttime);
}

/**
 * Sequential processing on the CPU
 **/
void on_cpu(long int *A, long int *B, long int *C,long int m, long int n){
	for(long int i = 0; i < m; i++){
        for(long int j = 0; j < n; j++){
            C[i*n+j] = (A[i*n + j] + B[j*m + i])*(B[j*m + i] - A[i*n + j]);
        }
    }
}

int main(int argc,char **argv){
    // variable declarations
    long int m,n;
    cin>>m>>n;

    // host_arrays
    long int *h_a,*h_b,*h_c;

    // device arrays
    long int *d_a,*d_b,*d_c;
    
    // Allocating space for the host_arrays
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));
    h_c = (long int *) malloc(m * n * sizeof(long int));

    // Allocating memory for the device arrays
    cudaMalloc(&d_a, m * n * sizeof(long int));
    cudaMalloc(&d_b, m * n * sizeof(long int));
    cudaMalloc(&d_c, m * n * sizeof(long int));

    // Read the input matrix A
    for(long int i = 0; i < m * n; i++) {
        cin>>h_a[i];
    }

    // Read the input matrix B
    for(long int i = 0; i < m * n; i++) {
        cin>>h_b[i];
    }

    // Transfer the input host arrays to the device
    cudaMemcpy(d_a, h_a, m * n * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, m * n * sizeof(long int), cudaMemcpyHostToDevice);

    long int gridDimx, gridDimy;
    
    // Launch the kernels
    /**
     * Kernel 1 - per_row_column_kernel
     * To be launched with 1D grid, 1D block
     **/
    gridDimx = ceil(float(m) / 1024);
    dim3 grid1(gridDimx,1,1);
    dim3 block1(1024,1,1);

    // double starttime = rtclock();  // -->
    per_row_column_kernel<<<grid1,block1>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();
    // double endtime = rtclock();  // -->
	  // printtime("GPU Kernel-1 time: ", starttime, endtime);  // -->

    cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m, n,"kernel1.txt");

    /**
     * Kernel 2 - per_column_row_kernel
     * To be launched with 1D grid, 2D block
     **/
    gridDimx = ceil(float(n) / 1024);
    dim3 grid2(gridDimx,1,1);
    dim3 block2(32,32,1);

    // starttime = rtclock();  // -->
    per_column_row_kernel<<<grid2,block2>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();
    // endtime = rtclock();  // -->
  	// printtime("GPU Kernel-2 time: ", starttime, endtime);  // -->

    cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m, n,"kernel2.txt");

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     **/
    gridDimx = ceil(float(n) / 16);
    gridDimy = ceil(float(m) / 64);
    dim3 grid3(gridDimx,gridDimy,1);
    dim3 block3(64,16,1);

    // starttime = rtclock();  // -->
    per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();
    // endtime = rtclock();  // -->
	  // printtime("GPU Kernel-3 time: ", starttime, endtime);  // -->

    cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m, n,"kernel3.txt");
 
    /**
     * Sequenctial CPU code
     **/
    // starttime = rtclock();  // -->
    // on_cpu(h_a,h_b,h_c,m,n);  // -->
    // endtime = rtclock();  // -->
	  // printtime("CPU time: ", starttime, endtime);  // -->

    return 0;
}

/**
 * Running the code for two pairs of (m,n) values on Colab GPU 
   gave the following results on one particular run:
 * (7863, 1121)
   GPU Kernel-1 time: 0.006352 seconds
   GPU Kernel-2 time: 0.063251 seconds
   GPU Kernel-3 time: 0.001880 seconds
   CPU time: 0.164905 seconds
 * (40,55)
   GPU Kernel-1 time: 0.000150 seconds
   GPU Kernel-2 time: 0.000047 seconds
   GPU Kernel-3 time: 0.000029 seconds
   CPU time: 0.000022 seconds
 * The following inference can be drawn for small matrices where 
   speed-up due to parallelism on a  GPU is compensated by lower 
   clock rates on  the GPU. Further to note, Kernel-1 takes more 
   time than Kernel-2 since m < n. 
 * On the CPU, the code is of O(mn), so for larger matrices, the
   speed-up due to  parallelism is farther more than in  case of 
   small matrices, while the  reduction in clock-rate is same in 
   both the cases. Thus, we see the exectuion time on the GPU is 
   lower than the case of CPU. Further, Kernel-1 finishes faster
   than Kernel-2 since m > n. 
 * Note that, this is only the time for computation and not data
   transfer between  host and device. To summarise, let x be the 
   speed-up due to parallel execution while  k, the reduction of 
   clock-rate. k is same for both the large and small data sizes
   while x is different, thus we  see the associated speed-up to 
   be more in case of large data sizes. 
 **/ 
   