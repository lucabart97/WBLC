/**
 * @file        cudaNeighbours.cpp
 * @author      Luca Bartoli(lucabartoli97@gmail.com)
 * @brief       Cuda class impelentation for get neighbours points for each point
 * @version     1.0
 * @date        2019-08-28
 * 
 * @copyright Copyright (c) 2019
 * 
 */
 #include <src/cudaNeighbours/cudaNeighbours.h>

__global__ void kernel(float *d_points, int* d_neighbours, int n, int neighbours, float eps, int MAXNEIGHB){

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    while(id < n){

        int i = 0;
        int s = id;
        
        for(int c = 0; c < neighbours/2; c++){

            s = (s+n-1) % n;

            float a     = d_points[id*4];
            float b     = d_points[s*4];
            float a_b   = a - b;
            float c     = d_points[id*4+1];
            float d     = d_points[s*4+1];
            float c_d   = c - d;

            if( (a_b*a_b + c_d*c_d) < eps){

                d_neighbours[MAXNEIGHB * id + i] = s;
                i++;
            }

            if( i > MAXNEIGHB/2-1){
                break;
            }
        }

        s = id;

        for(int c = 0; c < neighbours/2; c++){

            s = (s+1) % n;

            float a     = d_points[id*4];
            float b     = d_points[s*4];
            float a_b   = a - b;
            float c     = d_points[id*4+1];
            float d     = d_points[s*4+1];
            float c_d   = c - d;

            if( (a_b*a_b + c_d*c_d) < eps){

                d_neighbours[MAXNEIGHB * id + i] = s;
                i++;
            }

            if(i > MAXNEIGHB-2){
                break;
            }
        }

        d_neighbours[MAXNEIGHB * id + i] = -1;

        id += blockDim.x * gridDim.x;
    }
}



bool cudaNeighbours::init(int neighbours, float eps, int MAXNEIGHB,int cudaDevice){
    
    //Get device propriety
    ///////////////////////////////////////////////////////////

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount <= cudaDevice){
        std::cerr<<"cudaFiltering: Wrong GPU device ID\n";
        exit(-1);
    }

    cudaSetDevice(cudaDevice);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,cudaDevice);
    this->sm = deviceProp.multiProcessorCount;

    std::cout<<"Using: "<<deviceProp.name<<"\n\n";

    //////////////////////////////////////////////////////////


    this->eps           = eps;
    this->neighbours    = neighbours;
    this->MAXNEIGHB     = MAXNEIGHB;

    #ifndef COMPILE_FOR_NVIDIA_TX2

    //malloc on host (pinned memory)
    HANDLE_ERROR( cudaMallocHost((void**)&h_neighbours, MAXNEIGHB * MAXPOINTS * sizeof(int)) );
    #else

    #pragma message "Select compilation for TX2"
    h_neighbours = new int[MAXNEIGHB * MAXPOINTS];
    #endif

    HANDLE_ERROR( cudaMalloc((void**)&d_points,MAXPOINTS * sizeof(float) * 4) );
    HANDLE_ERROR( cudaMalloc((void**)&d_neighbours,MAXNEIGHB * MAXPOINTS * sizeof(int)) );

    return true;
}

void cudaNeighbours::calculateNeighbours(float *mat, int n){
    

    HANDLE_ERROR(   cudaMemset(d_neighbours,0,MAXNEIGHB * MAXPOINTS * sizeof(int))                                  );
    HANDLE_ERROR(   cudaMemcpy(d_points, mat, n * sizeof(float) * 4, cudaMemcpyHostToDevice)                        );

    kernel<<<this->sm,1024>>>(d_points, d_neighbours, n, this->neighbours, this->eps, MAXNEIGHB);

    HANDLE_ERROR(   cudaDeviceSynchronize()                                                                         );  
    HANDLE_ERROR(   cudaMemcpy(h_neighbours, d_neighbours , MAXNEIGHB * n * sizeof(int), cudaMemcpyDeviceToHost)    );
}

void cudaNeighbours::close(){

    cudaFree(d_points);

    #ifndef COMPILE_FOR_NVIDIA_TX2

    cudaFree(d_neighbours);
    #else

    delete [] h_neighbours;
    #endif
    
    cudaFree(h_neighbours);
}
