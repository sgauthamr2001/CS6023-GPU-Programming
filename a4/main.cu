/**
 * Title: CS6023, GPU Programming, Jan-May 2022, Assignment-4
 * Author: Sai Gautham Ravipati (EE19B053) 
 * Description: Allocation of train tickets to batch requests
 **/

#include <stdio.h>
#include <cuda.h>

using namespace std;

/**
 * Performs allocation of seats for a given batch of requests
 * The input data being used includes: 
    - R -> Number of requests in the batch 
    - req_src -> The source station of requests
    - req_dst -> The destination station of requests
    - req_tkt -> The number of tickets per request 
    - req_trn -> Train number of the requests 
    - req_cls -> Class number of the requests 
    - track -> Number of seats booked for train/class in each station
    - dst -> Final destination of the train 
    - src -> Starting station of the train 
    - cap -> Seat capacity of a given class in train 
    - thread -> Maintains map between request to thread 
    - max -> Last thread in the grid which does processing 
 * The output data includes: 
    - req_stat -> Status of each request, 1 for success, 0 for fail 
    - count[0] -> Total number of successes
    - count[1] -> Total number of failures
    - count[2] -> Total number of seats booked 
**/

__global__ void batch_kernel(
    int R, int *req_src, int *req_dst, int *req_tkt, int *req_trn, int *req_cls, 
    int *track, int *dst, int *src, int *cap, int *thread, int max, 
    unsigned int *req_stat, unsigned int *count){

    // Maintains a linked array of requests to be  processed by a
    // given thread. For say, consider requests 0, 1, 9, 2000 all
    // of them will be processed by thread 0. So link[0] shall be 
    // 1, link[1] holds 9 & link[9] holds 2000. By this approach, 
    // requests on the same thread are sequentialised while those
    // on different thread shall execute in parallel. 5000 is the 
    // maximum possible number of requests. 
    __shared__ int link[5000];    

    // Global id corresponding to request ID
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; 
    
    // Thread local variable ctr shall maintain the number of req
    // to be processed by a given thread. 
    int ctr = 0;

    // Accumulating the link array starting from the 1st request 
    int prev;
    for(int i = 0; i < R; i++){
        if(thread[i] == id){
            if(ctr == 0)               // Different processing for first request processed by thread
                prev = id;
            else{
                link[prev] = i;        // Updating the link of previous request to req. id of next request 
                prev = i;           
            }
            ctr += 1;                  // Accumulating the counter helps in traversing the array 
        }
    }

    // Only those threads which have id's less than the last occupied thread will execute 
    if(id <= max){
        int i = id;                          // The first task processed by a thread is its global id  
        for(int k = 0; k < ctr; k++){
            if(thread[i] == id){
                int fail = 0; 
                int trn = req_trn[i];        // Train of the given request 
                int cls = req_cls[i];        // Class of the given request 
                int stx, etx, d;             // Markers needed to check [src > dst] or [dst > src]

                // By using markers, common piece of code for both directions, reeduces number of 
                // instructions between the if, else statements, hence thread divergence reduces 
                if(dst[trn] > src[trn]){ 
                    stx = req_src[i];        // We check for tickets starting from source station 
                    etx = req_dst[i];        // Checking is completed if destination is reached
                    d = 1;                   // Offset is taken from source, d * (req_src - src), 1 to get +ve idx
                }
                else{
                    stx = req_dst[i] + 1;    // We start from req_dst + 1 and traverse as req_dst + 2, ...
                    etx = req_src[i] + 1;    // Stopped after reaching req_src
                    d = -1;                  // Offset from source, d * (req_src - src), -1 to get +ve idx
                }

                // Checking if seats are vacant and updating the flag fail
                for(int j = stx; j < etx; j++){
                    int seat_lt = cap[25 * trn + cls]; 
                    if(track[1250 * trn + 50 * cls + d * (j - src[trn])] + req_tkt[i] > seat_lt){
                        fail = 1;                       // Request fails if seats exceed in any of the station 
                        req_stat[i] = 0;                // 0 indicates a failure 
                        atomicInc(&count[0], 5000);     // Incrementing the fail-count, done by multiple threads
                        break; 
                    }
                } 

                // Seats are booked only if there are vacancies in all the stations 
                if(!fail){
                    for(int j = stx; j < etx; j++){     // Incrementing the booked seats 
                        track[1250 * trn + 50 * cls + d * (j - src[trn])] += req_tkt[i];
                    } 
                    req_stat[i] = 1;                    // 1 indicates a success
                    atomicInc(&count[1], 5000);         // Incrementing the pass-count and number of seats booked 
                    atomicAdd(&count[2], req_tkt[i] * (req_dst[i] - req_src[i]) * d);
                }
                i = link[i];     // Going to next request using the linked array till we reach counter 
            }
        }   
    }
}

int main(int argc,char **argv)
{
    int N;                     // Number of trains 
    scanf("%d", &N);           // Taking the number of trains from input

    int *src_arr = (int *) malloc (N * sizeof(int));              // Holds source station of trains
    int *dst_arr = (int *) malloc (N * sizeof(int));              // Holds destination station of trains  
    int *cap_arr = (int *) malloc (N * 25 * sizeof(int));         // Holds the capacity of each class
    int *thread  = (int *) malloc (N * 25 * sizeof(int));         // Holds the map of request to thread in each iteration

    for(int i = 0; i < N; i++){
        int trn_num;                                              // Train number 
        int cls_cap;                                              // Number of classes in the train   
        scanf("%d", &trn_num);                                    // Reading the train number 
        scanf("%d", &cls_cap);                                    // Reading the number of classes
        scanf("%d", &src_arr[trn_num]);                           // Reading source station of given train 
        scanf("%d", &dst_arr[trn_num]);                           // Reading destination station of given train 
        for(int j = 0; j < cls_cap; j++){
            int cls;
            scanf("%d", &cls);                                    // Reading the class number in a train 
            scanf("%d", &cap_arr[25 * trn_num + cls]);            // Reading the capacity of each class 
        }
    }

    int B;                // Number of batches 
    scanf("%d", &B);      // Reading the number of batches 

    int *req_trn = (int *) malloc (5000 * sizeof(int));           // Holds train number of each request 
    int *req_cls = (int *) malloc (5000 * sizeof(int));           // Holds class number of each request  
    int *req_src = (int *) malloc (5000 * sizeof(int));           // Holds the source of request 
    int *req_dst = (int *) malloc (5000 * sizeof(int));           // Holds the destination of request 
    int *req_tkt = (int *) malloc (5000 * sizeof(int));           // Holds the number of tickets per request

    int *map = (int *) malloc (5000 * sizeof(int));        // CPU array to store the map of request to threads
    int *stat = (int *) malloc (5000 * sizeof(int));       // Holds the status of each request on the CPU   
    int count[3];                                          // count[0] - num_pass, count[1] - num_fail, count[2] - num_tickets

    // Device arrays to hold source, destination, class capacity of trains
    int *d_dst, *d_src, *d_cap;     

    //  Allocating memory on device 
    cudaMalloc(&d_dst, N * sizeof(int));
    cudaMalloc(&d_src, N * sizeof(int));
    cudaMalloc(&d_cap, N * 25 * sizeof(int));

    // Copying data from host to device 
    cudaMemcpy(d_dst, dst_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, src_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, cap_arr, N * 25 * sizeof(int), cudaMemcpyHostToDevice);

    // Device arrays to hold request information 
    int *dreq_trn, *dreq_src, *dreq_dst, *dreq_tkt, *dreq_cls, *dmap;  

    // Allocating memory on device 
    cudaMalloc(&dreq_trn, 5000 * sizeof(int));
    cudaMalloc(&dreq_src, 5000 * sizeof(int));
    cudaMalloc(&dreq_dst, 5000 * sizeof(int));
    cudaMalloc(&dreq_tkt, 5000 * sizeof(int));
    cudaMalloc(&dreq_cls, 5000 * sizeof(int));
    cudaMalloc(&dmap, 5000 * sizeof(int));

    // Device arrays to hold the request status as well as counters 
    unsigned int *d_stat, *d_count;

    // Allocating memory on device 
    cudaMalloc(&d_stat, 5000 * sizeof(int));
    cudaMalloc(&d_count, 3 * sizeof(int));

    // Device array to hold track of number of seats booked in each station 
    int *d_track; 

    // Allocating memory on device 
    cudaMalloc(&d_track, N * 25 * 50 * sizeof(int));
     
    // Setting the number of seats booked to 0 initially 
    cudaMemset(d_track, 0, N * 25 * 50 * sizeof(int));
    cudaDeviceSynchronize(); 

    int batch_size;         // Batch-size in each iteration 
    int req_id;             // Holds the request id 
    int max;                // Holds the last request going to different train and class

    for (int i = 0; i < B; i++){
        memset(thread, -1, N * 25 * sizeof(int));        // Setting the map to -1 at the start of each iteration
        scanf("%d", &batch_size);                        // Getting the input batch size
        for(int j = 0; j < batch_size; j++){              
            scanf("%d", &req_id);                        // Reading the request id
            scanf("%d", &req_trn[req_id]);               // Reading the request train 
            scanf("%d", &req_cls[req_id]);               // Reading the request class
            scanf("%d", &req_src[req_id]);               // Reading the request source 
            scanf("%d", &req_dst[req_id]);               // Reading the request destination
            scanf("%d", &req_tkt[req_id]);               // Reading the number of tickets required 

            int trn = req_trn[req_id];               // Train of given request
            int cls = req_cls[req_id];               // Class of given request 

            // Getting the map of request to thread while processing the input reads
            // If a given train & class is not assigned to any of threads, the least 
            // request id shall be the thread id. If already assinged then that req. 
            // goes to the thread id assigned to the train and class. 
            if(thread[25 * trn + cls] == -1){             
              thread[25 * trn + cls] = req_id;            // Case of given train and class coming first time 
              map[req_id] = req_id;                  
              max = req_id;                               // Needed for appropriate threads processing the req.
            }
            else
              map[req_id] =  thread[25 * trn + cls];      // Case of thread already assigned to a train & class
        }

        // Copying of request data from host to device
        cudaMemcpy(dreq_trn, req_trn, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreq_src, req_src, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreq_dst, req_dst, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreq_tkt, req_tkt, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dreq_cls, req_cls, batch_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dmap, map, batch_size * sizeof(int), cudaMemcpyHostToDevice); 

        // Setting the output results to zero initially  
        cudaMemset(d_stat, 0, batch_size * sizeof(int));
        cudaMemset(d_count, 0, 3 * sizeof(int));
        cudaDeviceSynchronize(); 

        // Launch configuration of the kernel 
	      int gridDimx = ceil(float(batch_size)/1024);
	      int blockDimx = 1024; 

        // Launching the kernel to process this batch of requests  
	      batch_kernel<<<gridDimx, blockDimx>>>(
        batch_size, dreq_src, dreq_dst, dreq_tkt, dreq_trn, dreq_cls,
        d_track, d_dst, d_src, d_cap, dmap, max, d_stat, d_count);

        // Copying the results obtained back to the CPU
        cudaMemcpy(stat, d_stat, batch_size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(count, d_count, 3 * sizeof(int), cudaMemcpyDeviceToHost);

        // Printing the results in desired format to the output 
        for(int j = 0; j < batch_size; j++ ){
            if(stat[j] == 1)
              printf("success\n");
            else
              printf("failure\n");
        } 
        printf("%d %d\n", count[1], count[0]);
        printf("%d\n", count[2]);
    }

    // Freeing the device arrays 
    cudaFree(d_dst);
    cudaFree(d_src);
    cudaFree(d_cap);
    cudaFree(dreq_trn);
    cudaFree(dreq_src);
    cudaFree(dreq_dst);
    cudaFree(dreq_tkt);
    cudaFree(dreq_cls);
    cudaFree(dmap); 
    cudaFree(d_stat);
    cudaFree(d_count);
    cudaFree(d_track);

    // Freeing the host memories 
    free(src_arr); 
    free(dst_arr); 
    free(cap_arr);
    free(thread);
    free(req_trn);
    free(req_cls);
    free(req_src);
    free(req_dst);
    free(req_tkt); 
    free(stat);  
    free(map);

    return 0;
}

/* End of Code */ 
