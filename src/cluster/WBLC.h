#pragma once
/**
 * @file        WBLC.h
 * @author      Luca Bartoli(lucabartoli97@gmail.com)
 * @brief       File that contains the WBLC class definition
 * @version     1.0
 * @date        2019-08-28
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include <ctime>
#include <iostream>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <src/cudaNeighbours/cudaNeighbours.h>
#include <omp.h>

class WBLC{
    private:

        /**********************************************************************************************************************/
        static const int        MAXCLUSTER              = 1000;
        static const int        MAXPOINTS               = 200000;
        static const int        MAXDELETE               = 200;
        int                     NEARPOINTS              = 0;
        double                  eps;
        int                     minDimCluster;
        int                     nearPoints;
        int                     queue[MAXPOINTS];
        int                     queueElements;
        Eigen::MatrixXf*        cloud;
        int                     cloudDim;
        float                   x,y,z;
        int                     t;
        bool                    read[MAXPOINTS];
        cudaNeighbours          neigh;
        int                     deleteQueue[MAXDELETE];
        int                     clusterLenght;
        /**********************************************************************************************************************/


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief               Metodo per trovare i punti vicini 
         * 
         * @param id            Id del punto di cui analizzare i vicini
         * 
         */
        void                    neighbourPoints(int id);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    public:

        
        /**********************************************************************************************************************/
        
        /**
         * @brief Array that contain in which cluster is point in the same position respect to matrix points passed to WBLC.
         *          -1 if point is in no clusters
         * 
         */
        int                                 clusterPoint[MAXPOINTS];

        /**
         * @brief Number of detected clusters
         * 
         */
        int                                 NCluster;
        /**********************************************************************************************************************/


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief               Method that cluster lidar points
         * 
         * @param eps           Maximum distance between 2 point for have they in the same cluster
         * @param minDimCluster Dimension of minimun cluster
         * @param cudaDevice    GPU device ID (default: 0)
         * @return true         init ok
         * @return false        not init ok
         */
        bool                                init(float eps, int minDimCluster, int WINDOW, int MAXNEIGHB, int cudaDevice = 0);

        /**
         * @brief               Method that calculate the clusters
         * 
         * @param points        Lidar points
         *                          Input format:
         *                                          (X)
         *                                          (Y)
         *                                          (Z) * dim
         *                                          (I)
         * 
         * @param dim           Number of points
         * 
         * @return BoundingBox  Matrix contain bounding box information
         */
        void                                clustering(Eigen::MatrixXf* points, int dim);

        /**
         * @brief               Method that clean memory
         * 
         */
        void                                close();
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};