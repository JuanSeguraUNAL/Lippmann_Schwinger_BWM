#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include "ConfocalParabolicBilliard.hpp"
#include "BoundaryWallMethod.hpp"

int main(int argc, char* argv[]){
    if(argc<6){
        std::cerr<<"Uso: "<<argv[0]<<" xi0 eta0 k2\n";
        return 1;
    }
    double xi0   = std::stod(argv[1]);
    double eta0  = std::stod(argv[2]);
    double k2    = std::stod(argv[3]);
    double angle = std::stod(argv[4]);
    int    threads = std::atoi(argv[5]);
    double gamma = std::numeric_limits<double>::infinity();
    int    num_b = std::stod(argv[6]);
    int    Ngrid = num_b;

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    // 1) frontera
    bwm::ConfocalParabolicBilliard billiard(xi0, eta0, num_b);
    auto boundary = billiard.getBoundary();
    std::ofstream foutB("boundary.dat");
    for(auto &p: boundary) foutB<<p.x()<<" "<<p.y()<<"\n";
    foutB.close();

    // 2) solver BWM
    std::vector<double> gvec{gamma};
    bwm::BoundaryWallMethod solver(boundary, gvec, std::sqrt(k2), angle, threads);

    // 3) malla y volcado de density.dat y phase.dat
    start = std::chrono::high_resolution_clock::now();
    double xmin=-8, xmax=8;
    for(int i=0;i<Ngrid;++i){
        double y = xmin + i*(xmax-xmin)/(Ngrid-1);
        for(int j=0;j<Ngrid;++j){
            double x   = xmin + j*(xmax-xmin)/(Ngrid-1);
            auto ψ     = solver.computeScatteredWave({{x,y}})[0];
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    std::cout << num_b << "\t" << elapsed.count() << "\n";

    return 0;
}