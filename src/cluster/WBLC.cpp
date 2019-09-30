/**
 * @file        WBLC.cpp
 * @author      Luca Bartoli(lucabartoli97@gmail.com)
 * @brief       File that contains the WBLC class implementation
 * @version     1.0
 * @date        2019-08-28
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include <src/cluster/WBLC.h>

bool WBLC::init(float eps, int minDimCluster, int WINDOW, int MAXNEIGHB, int cudaDevice){

    this->eps           = eps*eps;
    this->minDimCluster = minDimCluster;
    this->NEARPOINTS    = WINDOW;

    std::memset(read,1,MAXPOINTS);
    neigh.init(NEARPOINTS,this->eps, MAXNEIGHB, cudaDevice);

    return true;
}

void WBLC::clustering(Eigen::MatrixXf* points, int dim){

    this->cloud         = points;
    this->cloudDim      = dim;
    this->NCluster      = 0;
    this->queueElements = 0;

    this->neigh.calculateNeighbours(this->cloud->data(),this->cloudDim);

    for(int i = 0; i < this->cloudDim; i++){        

        this->clusterLenght = 1;

        if(read[i]){

            if(this->NCluster == MAXCLUSTER){

                std::cerr<<"cluster: Maximum number of detectable clusters\n";
                break;
            }

            read[i] = false;

            clusterPoint[i] = NCluster;

            this->queue[this->queueElements] = i;
            this->queueElements++;

            deleteQueue[0] = i;

            while(this->queueElements != 0){

                this->queueElements--;

                neighbourPoints(this->queue[this->queueElements]);
            }

            if(this->clusterLenght  > this->minDimCluster){

                this->NCluster++;
            }else{

                for(int c=0; c < this->clusterLenght; c++){

                    int pos             = deleteQueue[c];
                    clusterPoint[pos]   = -1;
                }
            }
        }
    }

    std::memset(read,1,this->cloudDim);
}

void WBLC::neighbourPoints(int id){

    id *= this->neigh.MAXNEIGHB;

    while(this->neigh.h_neighbours[id] != -1){

        t = this->neigh.h_neighbours[id];

        if( read[t] ){

            read[t] = false;

            this->queue[this->queueElements] = t;
            this->queueElements++;

            clusterPoint[t] = NCluster;
            this->clusterLenght++;

            if(this->clusterLenght < this->minDimCluster){

                deleteQueue[this->clusterLenght-1]  = t;
            }

        }

        id +=1;
    }
}

void WBLC::close(){

    neigh.close();
}