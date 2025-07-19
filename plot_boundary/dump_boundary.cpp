#include <iostream>
#include <cstdlib>        // for std::stod
#include "ConfocalParabolicBilliard.hpp"

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cerr << "Uso: " << argv[0] << " xi0 eta0\n";
        return 1;
    }
    
    double xi0  = std::stod(argv[1]);
    double eta0 = std::stod(argv[2]);
    int num_points = 200; 
    bwm::ConfocalParabolicBilliard billiard(xi0, eta0, num_points);
    for(const auto& p : billiard.getBoundary())
        std::cout << p.x() << " " << p.y() << "\n";

    return 0;
}
