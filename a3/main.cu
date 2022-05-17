/**
 * Title: CS6023, GPU Programming, Jan-May 2022, Assignment-3
 * Author: Sai Gautham Ravipati (EE19B053) 
 * Description: Multi-core Task Scheduling

 * Modified elements: 
    - schedule_kernel: Calculates the turn-around time for tasks.
	  - operations: Performs mem. alloc. and calls the kernels.
 **/

#include <stdio.h>
#include <cuda.h>

using namespace std;

__global__ void schedule_kernel(int m, int n, int *executionTime, int *priority, int *result){
    
    // Variable declerations for task handling 
    __shared__ int task_ctr;          // Keeps track of the current task
    __shared__ int core_cache;        // Implements blocking between succesive tasks on queue     
    __shared__ int core_ctr[1024];    // Keeps track of running time on each core 
    __shared__ int core_pin[1024];    // Maintains priority to core mapping 

    __shared__ int mode;              // mode -> 1, Assigns core for a new priority
    __shared__ int comp_val[1024];    // Used in reductively finding free core 
    __shared__ int comp_idx[1024];    // Stores core_ids for reduction 
    
    int idx = threadIdx.x;    // Shall be same as number of cores
    core_ctr[idx] =  0;       // Init. counter/core to zero parallely 
    core_pin[idx] = -1;       // Init. priority to core mapping to -1 

    if(idx == 0){             // Only thread_0 sets shared variables  
        mode = 0;             // Init. mode -> 0, set to 1 if core_pin[i] = -1
        task_ctr = 0;         // Init. task_ctr to 0
        core_cache = 0;       // Init. core_cache to 0 
    }
    __syncthreads();          // Needed for init. to be visible to all threads
    
    while(task_ctr < n){
        if(core_pin[priority[task_ctr]] == -1){                    // Checking if a priority is coming for first time
            if(priority[task_ctr] == idx){                         // Thread same as priority updates mode 
                mode = 1;                                          // Setting mode to 1, used to find core
                comp_val[m] = max(core_ctr[idx] , core_cache);     // Padding for reduction with odd no. of items
                comp_idx[m] = idx;                                 
            }
        }
        __syncthreads();    // Needed for mode to be visible to all threads

        if(mode){
            comp_val[idx] = max(core_ctr[idx] , core_cache);    // Storing values to comp_val, comp_idx
            comp_idx[idx] = idx;                                // Reduction is applied on these 2 
            __syncthreads();                                    // Needed for data write to be completed 

            // Reductively finding the free core
            for(int off = ceil(float(m)/2); off > 1; off = ceil(float(off)/2)){
                if(idx < off){
                    if(comp_val[idx] > comp_val[idx + off]){
                        comp_val[idx] = comp_val[idx + off]; 
                        comp_idx[idx] = comp_idx[idx + off]; 
                    }
                }
                __syncthreads();
            }

            if(priority[task_ctr] == idx){                 // Thread same as priority updates the core mapping 
                core_pin[idx] = (comp_val[0] > comp_val[1]) ? comp_idx[1] : comp_idx[0];
                mode = 0;                                  // Mode is set back to zero 
            }
        }

        // Task is processed by thread same as its priority
        if(priority[task_ctr] == idx){                     
            int core_idx = core_pin[idx];

            if(core_ctr[core_idx] > core_cache)
                core_cache = core_ctr[core_idx];
            
            core_ctr[core_idx] = core_cache + executionTime[task_ctr]; 
            result[task_ctr] = core_ctr[core_idx];         // Storing the turn-around times in result 

            task_ctr++;    // Incrementing the task counter to process next task 
        }
        __syncthreads();   
    }
}

//Complete the following function
void operations(int m, int n, int *executionTime, int *priority, int *result){
	  int *d_executionTime, *d_priority, *d_result; 
	
  	// allocate memory...
  	// Input and output vectors
  	cudaMalloc(&d_executionTime, n * sizeof(int));
  	cudaMalloc(&d_priority, n * sizeof(int));
  	cudaMalloc(&d_result, n * sizeof(int));

    // copy the values...
    cudaMemcpy(d_executionTime, executionTime, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_priority, priority, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // call the kernels for doing required computations...
    // Kernel launch to schedule the tasks, no. of threads same as no. of cores
    schedule_kernel<<<1 , m>>>(m, n, d_executionTime, d_priority, d_result);

    // copy the result back...
    cudaMemcpy(result, d_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    // deallocate the memory...
    cudaFree(d_executionTime);
    cudaFree(d_priority);
    cudaFree(d_result);
}

int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
    //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================


	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
    free(executionTime);
    free(priority);
    free(result);
    
    
    
}


