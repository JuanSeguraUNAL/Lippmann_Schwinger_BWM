#include <iostream>
#include <cstdlib>        // for std::stod
#include "ConfocalParabolicBilliard.cuh"

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cerr << "Uso: " << argv[0] << " xi0 eta0\n";
        return 1;
    }

    double xi0 = std::stod(argv[1]);
    double eta0 = std::stod(argv[2]);
    int num_points = 200;
    bwm::ConfocalParabolicBilliard billiard(xi0, eta0, num_points);
    Point* boundary = billiard.getBoundary();
    for(int i = 0; i < num_points; i++){
        std::cout << boundary[i].x << "\t" << boundary[i].y << "\n";
    }

    free(boundary);

    return 0;
}