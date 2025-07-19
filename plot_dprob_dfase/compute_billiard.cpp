#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include "ConfocalParabolicBilliard.hpp"
#include "BoundaryWallMethod.hpp"

int main(int argc, char* argv[]){
    if(argc<4){
        std::cerr<<"Uso: "<<argv[0]<<" xi0 eta0 k2\n";
        return 1;
    }
    double xi0   = std::stod(argv[1]);
    double eta0  = std::stod(argv[2]);
    double k2    = std::stod(argv[3]);
    double angle = 0.0;
    double gamma = std::numeric_limits<double>::infinity();
    int    num_b = 200;
    int    Ngrid = 200;

    // 1) frontera
    bwm::ConfocalParabolicBilliard billiard(xi0, eta0, num_b);
    auto boundary = billiard.getBoundary();
    std::ofstream foutB("boundary.dat");
    for(auto &p: boundary) foutB<<p.x()<<" "<<p.y()<<"\n";
    foutB.close();

    // 2) solver BWM
    std::vector<double> gvec{gamma};
    bwm::BoundaryWallMethod solver(boundary, gvec, std::sqrt(k2), angle);

    // 3) malla y volcado de density.dat y phase.dat
    std::ofstream foutD("density.dat"), foutP("phase.dat");
    double xmin=-8, xmax=8;
    for(int i=0;i<Ngrid;++i){
        double y = xmin + i*(xmax-xmin)/(Ngrid-1);
        for(int j=0;j<Ngrid;++j){
            double x   = xmin + j*(xmax-xmin)/(Ngrid-1);
            auto ψ     = solver.computeScatteredWave({{x,y}})[0];
            foutD<<std::norm(ψ)<<(j+1==Ngrid? "\n":" ");
            foutP<<std::arg(ψ)<<(j+1==Ngrid? "\n":" ");
        }
    }
    return 0;
}

