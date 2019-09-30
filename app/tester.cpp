/**
 * @file        tester.cpp
 * @author      Luca Bartoli(lucabartoli97@gmail.com)
 * @brief       Example for testing the clustering
 * @version     1.0
 * @date        2019-08-28
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include <string>
#include <pthread.h>
#include <mutex>
#include <ctime>
#include <csignal>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <src/cluster/WBLC.h>

//Thread status
bool gRun = true;

//Params from yaml file
std::string path;
float maxD;
int minCluster, W, maxN;


//Signal handler
void signal_handler(int signal)
{
    gRun = false;
    std::cout<<"\nRequest closing..\n";
}


//Function that read dataset; data format: X Y Z I
//Dataset lidar height: 1.60
bool readFromTk(int n, Eigen::MatrixXf& cloud){

    std::string file = path + "cloud"+std::to_string(n)+".bin";
    std::ifstream is(file.c_str());
    if(!is)
        return false;

    int size[2] = { 0, 0 };
    is.read((char*)size, 2*sizeof(int));
    cloud.resize(size[0], size[1]);
    is.read((char *)cloud.data(), cloud.size() * sizeof(float));
    return true;
}

//Easy filter that delete ground and tree using z. 
//For cluster the car, pedestrian and truck, we take the points from 0 to 2.25
void filter(Eigen::MatrixXf& cloud){

    int points_filtered=0;

    for(int s = 0; s < cloud.cols(); s++){

        float x = cloud(0,s);
        float y = cloud(1,s);
        float z = cloud(2,s);
        float i = cloud(3,s);

        if(z > -1.25 && z < 1 && sqrt(x*x+y*y) < 50){
                cloud(0,points_filtered) = x;
                cloud(1,points_filtered) = y;
                cloud(2,points_filtered) = z + 1.25;
                cloud(3,points_filtered) = i;
                points_filtered++;
        } 
    }

    cloud.conservativeResize(4,points_filtered);
}

//Clustering thread with time calculation
void* clustering(void*){

    Eigen::MatrixXf cloud;
    cloud.resize(4,150000);

    int file_name=1;

    WBLC cls;
    cls.init(maxD,minCluster, W, maxN);


    //Timing
    //////////////////////////////////
    double min = 9999;
    double max = -1;
    long double avg_time = 0;;
    long long int avg_point = 0;
    long int scan = 0;
    //////////////////////////////////

    while(readFromTk(file_name,cloud) && gRun){

        file_name++;
        filter(cloud);

        //Timing
        //////////////////////////////////
        timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        //////////////////////////////////

        cls.clustering(&cloud,cloud.cols());

        //Timing
        //////////////////////////////////
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time = ((double)(end.tv_sec - start.tv_sec) * 1.0e9 + (double)(end.tv_nsec - start.tv_nsec))/1.0e6;
        scan ++;
        avg_time += time;
        if(time < min)
            min = time;
        if(time > max)
            max = time;
        //////////////////////////////////

        std::cout<<"Find "<<cls.NCluster<<" clusters\t->\t"<<time<<"\tms"<<std::endl;
    }

    std::cout<<"\n\nRESULTS:\n";
    std::cout<<"\tMax  : "<<max<<"\n";
    std::cout<<"\tMin  : "<<min<<"\n";
    std::cout<<"\tAvg  : "<<avg_time/scan<<"\n";
    std::cout<<"\tScan : "<<scan<<"\n";
    cls.close();
    pthread_exit(NULL);
}


int main(int argc, char* argv[]){

    YAML::Node config   = YAML::LoadFile("../conf/config.yaml");
    maxD                = config["maxD"].as<float>();
    maxN                = config["maxN"].as<int>();
    W                   = config["W"].as<int>();
    minCluster          = config["minCluster"].as<int>();
    path                = config["path"].as<std::string>();

    std::cout<<"\tpath:\t\t"<<path<<"\n\tminCluster:\t"<<minCluster<<"\n\tmaxN:\t\t"<<maxN<<"\n\tmaxD:\t\t"<<maxD<<"\n\n";

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    pthread_t t1;
    pthread_create(&t1, NULL, clustering, NULL);
    pthread_join(t1,NULL);

    return 0;

}