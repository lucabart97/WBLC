#pragma once
/**
 * @file        cudaNeighbours.h
 * @author      Luca Bartoli(lucabartoli97@gmail.com)
 * @brief       Cuda class definition for get neighbours points for each point
 * @version     1.0
 * @date        2019-08-28
 * 
 * @copyright Copyright (c) 2019
 * 
 */
//#define COMPILE_FOR_NVIDIA_TX2
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


class cudaNeighbours{

    public:

        static const int    MAXPOINTS       = 50000;
        int                 MAXNEIGHB       = 0;
        int*                h_neighbours;


    private:

        int                 neighbours;
        float               eps;
        float               *d_points;
        int                 *d_neighbours;
        int                 sm;
    
    public:

        /**
         * @brief               init method
         * 
         * @param neighbours    Neighbours search number
         * @param eps           Max distance between 2 points
         * @param cudaDevice    GPU device ID
         * @return true         init success
         * @return false        init abort
         */
        bool                init(int neighbours, float eps, int MAXNEIGHB, int cudaDevice);

        /**
         * @brief               Method that calculate the neighbours for each point
         * 
         * @param mat           Matrix that contains the points. Format: XYZIXYZI...
         * @param n             Points number
         */
        void                calculateNeighbours(float *mat, int n);

        /**
         * @brief               Method that clean memory
         * 
         */
        void                close();

};

__global__ void kernel(float *d_points, int* d_neighbours, int n, int neighbours, float eps, int MAXNEIGHB);