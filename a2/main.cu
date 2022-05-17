/**
 * Title: CS6023, GPU Programming, Jan-May 2022, Assignment-2
 * Author: Sai Gautham Ravipati (EE19B053) 
 * Description: Computation of a matrix X = (A + B.T)(C)(D.T)

 * Modified elements: 
    - compute_stage_1_kernel: Process each element of A + B.T.
	- compute_stage_2_kernel: Matrix product of (A + B.T) & C.
	- compute_stage_3_kernel: Process each element of X.
	- compute: Performs mem. alloc. and calls the kernels.

 * Procedure adopted: 
	- All the matrices are processed as a 32 x 32 tile, to get 
	  better coalescing, and bank conflicts are kept minimised 
	  in case of shared memory.  
	- The computation of X is broken down into 3 computations,
	  (A + B.T) is computed and is stored in temp1 (p,q), then
	  matrix multiplication of temp1,C is performed, result is
	  stored in temp2 (p,r). The final kernel computes X using 
	  both temp2 and D. 

 * References: 
	The documentation briefly describes on improved coalescing 
	for computations involving transpose matrices. 
	https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf
 **/

#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

// Kernels used for computation 

/**
 * The following kernel reads in A, B to compute (A + B.T) and
   has been tiled for better coalescing. Bank conflicts can be 
   avoided in case of B.T by slight modification of the shared 
   memory. The result is stored in global mem. temp1, which is 
   accessed by the next for computation. 
 * On a colab GPU, using tiling and shared memory, in one par-
   -ticular run, this takes 110.98 us for large test case, in 
   contrast to 368.45 us for the baseline version without any 
   usage of shared memory. 
 * Speed-up here beacuse of two fold, one being coalesced mem.
   access, as well as, avoiding bank conflicts. 
 * The launch configuration assumes blockDim.x,blockDim.y = 32
   and it is same as the tile size.  
 **/

__global__ void compute_stage_1_kernel(int *A, int *B, int p, int q, int *temp1){
	__shared__ int tile1[32][32];    // Stores elements of matrix A
	__shared__ int tile2[32][33];    // Stores elements of matrix B, 33 (dim. y) to avoid bank conflicts

	unsigned j1 = blockIdx.x * blockDim.x + threadIdx.x;    // id to access rows of A
	unsigned i1 = blockIdx.y * blockDim.x + threadIdx.y;    // id to access columns of A

	unsigned i2 = blockIdx.y * blockDim.x + threadIdx.x;    // id to access columns of B
	unsigned j2 = blockIdx.x * blockDim.x + threadIdx.y;    // id to access rows of B 

	// Filling of tiles happens in a coalesced manner, threadIdx.x doesn't have scale factor in j1, i2

	if(i1 < p && j1 < q)
		tile1[threadIdx.y][threadIdx.x] =  A[i1 * q + j1];    // Filling up tile of A
	else
		tile1[threadIdx.y][threadIdx.x] =  0;    // Checking for spilling of the tile

	if(i2 < p && j2 < q)    
		tile2[threadIdx.y][threadIdx.x] =  B[j2 * p + i2];    // Filling up tile of B 
	else
		tile2[threadIdx.y][threadIdx.x] =  0;    // Checking for spilling of the tile

	__syncthreads();    // Needed for proper access to tile2[threadIdx.x][threadIdx.y]  

	// Storing the output in temp1, used by next kernel, 33 (dim. y) avoids bank conflict for tile2

	if(i1 < p && j1 < q)
    	temp1[i1 * q + j1] =  tile1[threadIdx.y][threadIdx.x] + tile2[threadIdx.x][threadIdx.y]; 
}

/**
 * The kernel reads in temp1 and C in coalesced manner. To get
   output temp2, we fill in the same tile wise. Filling of one 
   tile of output, requires iteration of length (q/32), on the 
   tiles of temp2 and C, since this is a matrix multiplication
   in contrast to the previous case. Each thread accumulates a 
   sum variable, to store the product. The result is stored in 
   global mem. temp2, which is accessed by the next kernel. 
 * On a colab GPU, this takes 4.4172 ms on large test case, in
   comparision to that of baseline which took 19.009 ms, there
   is a speed-up mainly beacuase of coalesced accesses.  
 * The launch configuration assumes blockDim.x,blockDim.y = 32
   and it is same as the tile size as in the previous case.
 **/

__global__ void compute_stage_2_kernel(int *temp1, int *C, int p, int q, int r, int *temp2){
	__shared__ int tile1[32][32];    // Stores elements of temp1
	__shared__ int tile2[32][32];    // Stores elements of C

	unsigned j2 = blockIdx.x * blockDim.x + threadIdx.x;    // id to access columns of C
	unsigned i1 = blockIdx.y * blockDim.x + threadIdx.y;    // id to access rows of temp1

  	int tile_len = ceil(float(q)/32);    // Length of iteration where tiled products of temp1, C are accumulated
  	int sum = 0;                         // Final output value to be stored to temp2 

	for(int t = 0; t < tile_len; t++){
		int j1 = t * blockDim.x + threadIdx.x;    // id to access columns of temp1
    	int i2 = t * blockDim.x + threadIdx.y;    // id to access rows of C

		// Tiles are filled in a coalsced manner as threadIdx.x doesn't have scale factor in j1, i2

    	if(i1 < p && j1 < q)
			tile1[threadIdx.y][threadIdx.x] =  temp1[i1 * q + j1];    // Filling up tile of temp1 for that iteration
    	else
      		tile2[threadIdx.y][threadIdx.x] = 0;     // Checking for spilling of the tile  

    	if(i2 < q && j2 < r)
			tile2[threadIdx.y][threadIdx.x] = C[i2 * r + j2];    // Filling up tile of C for that iteration
    	else 
    		tile2[threadIdx.y][threadIdx.x] = 0;    // Checking for spilling of the tile  

		__syncthreads();    // Needed to accumulate the product 
			
		for(int k = 0; k < blockDim.x; k++){
			sum += tile1[threadIdx.y][k] * tile2[k][threadIdx.x];    // No bank conflict
		}

		__syncthreads();    // Ensures tile1, tile2 are unchanged till sum is accumulated
  	}

	// Storing the output in temp2, used by next kernel
	if(i1 < p && j2 < r)
    	temp2[i1 * r + j2] = sum;
}

/**
 * The kernel reads in temp2 and D in coalesced manner. To get
   output X, tiled multiplication is done as before. Similarly
   the tiled multiplication is iterated for (r/32) times since 
   number of columns (common dimension) over here is r and the 
   sum variable is accumulated.  The result is stored to X. 
 * On a colab GPU, this takes 2.3061 ms on large test case, in
   comparision to that of baseline which took 88.860 ms, there
   is a speed-up mainly beacuase of coalesced acesses, in this
   case, a transposed matrix, as well as, bank conflicts avoi-
   -ded.  
 * The launch configuration assumes blockDim.x,blockDim.y = 32
   and it is same as the tile size. 
 **/

__global__ void compute_stage_3_kernel(int *temp2, int *D, int p, int r, int s, int *X){	
	__shared__ int tile1[32][32];    // Stores elements of matrix temp2
	__shared__ int tile2[32][33];    // Stores elements of matrix D, 33 (dim. y) to avoid bank conflicts

	unsigned j2 = blockIdx.x * blockDim.x + threadIdx.x;    // id to access columns of X

	unsigned i1 = blockIdx.y * blockDim.x + threadIdx.y;    // id to access rows of temp2
	unsigned i2 = blockIdx.x * blockDim.x + threadIdx.y;    // id to access rows of D

	int tile_len = ceil(float(r)/32);    // Length of iteration, common dimension here is r 
	int sum = 0;                         // Final output value to be stored to X

	// Tiles are filled in a coalsced manner as threadIdx.x doesn't have scale factor in j1

	for(int t = 0; t < tile_len; t++){
		int j1 = t * blockDim.x + threadIdx.x;    // id to access columns of temp2, D to fill the tile 

		if(i1 < p && j1 < r)
			tile1[threadIdx.y][threadIdx.x] = temp2[i1 * r + j1];    // Filling up tile of temp2 for that iteration
		else
			tile1[threadIdx.y][threadIdx.x] = 0;    // Checking for spilling of the tile  
		if(i2 < s && j1 < r)
			tile2[threadIdx.y][threadIdx.x] = D[i2 * r + j1];    // Filling up tile of D for that iteration
		else 
			tile2[threadIdx.y][threadIdx.x] = 0;    // Checking for spilling of the tile  

		__syncthreads();    // Needed to accumulate the product 
			
		for(int k = 0; k < blockDim.x; k++){
			sum += tile1[threadIdx.y][k] * tile2[threadIdx.x][k];    // No bank conflict as dim. y (tile2) is 33 
		}
		__syncthreads();    // Ensures tile1, tile2 are unchanged till sum is accumulated
	}

	// Storing the output to X
	if(i1 < p && j2 < s)
    	X[i1 * s + j2] = sum;
}

// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX){
	// variable declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixX;
	int *temp1, *temp2;
	
	// allocate memory...
	// Input and output matrices
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * p * sizeof(int));
	cudaMalloc(&d_matrixC, q * r * sizeof(int));
	cudaMalloc(&d_matrixD, s * r * sizeof(int));
	cudaMalloc(&d_matrixX, p * s * sizeof(int));

	// Temporary matrices used between kernels 
	cudaMalloc(&temp1, p * q * sizeof(int));
	cudaMalloc(&temp2, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);

	
	// call the kernels for doing required computations...
	// Tile size is same as block dimensions, so both of them are 32 & shouldn't be changed 

	// Kernel-1 
	int gridDimx1 = ceil(float(q)/32);
	int gridDimy1 = ceil(float(p)/32);
	dim3 grid1(gridDimx1, gridDimy1, 1);
	dim3 block1(32, 32, 1);
	compute_stage_1_kernel<<<grid1, block1>>>(d_matrixA, d_matrixB, p, q, temp1);

	// Kernel-2 
	int gridDimx2 = ceil(float(r)/32);
	int gridDimy2 = ceil(float(p)/32);
	dim3 grid2(gridDimx2, gridDimy2,1);
	dim3 block2(32, 32, 1);
	compute_stage_2_kernel<<<grid2, block2>>>(temp1, d_matrixC, p, q, r, temp2);

	// Kernel-3
	int gridDimx3 = ceil(float(s)/32);
	int gridDimy3 = ceil(float(p)/32);
  	dim3 grid3(gridDimx3, gridDimy3, 1);
 	dim3 block3(32, 32, 1);
	compute_stage_3_kernel<<<grid3, block3>>>(temp2, d_matrixD, p, r, s, d_matrixX);

	// copy the result back...
	cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixX);
	cudaFree(temp1);
	cudaFree(temp2);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * p * sizeof(int));
	matrixC = (int*) malloc(q * r * sizeof(int));
	matrixD = (int*) malloc(s * r * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	matrixX = (int*) malloc(p * s * sizeof(int));

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixX);

	return 0;
}
	