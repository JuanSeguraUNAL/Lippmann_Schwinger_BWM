#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include "ConfocalParabolicBilliard.cuh"
#include "BoundaryWallMethod.cuh"

int main(int argc, char* argv[]){
    if(argc<6){
        std::cerr<<"Uso: "<<argv[0]<<" xi0 eta0 k2\n";
        return 1;
    }
    double xi0   = std::stod(argv[1]);
    double eta0  = std::stod(argv[2]);
    double k2    = std::stod(argv[3]);
    double angle = std::stod(argv[4]);
    double gamma = std::numeric_limits<double>::infinity();
    int    num_b = std::stod(argv[5]);
    int    Ngrid = num_b;

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> elapsed;

    // 1) Frontera
    bwm::ConfocalParabolicBilliard billiard(xi0, eta0, num_b);
    auto boundary = billiard.getBoundary();
    std::ofstream foutB("boundary.dat");
    for(int i = 0; i < num_b; i++){
        foutB << boundary[i].x << "\t" << boundary[i].y << "\n";
    }
    foutB.close();
    cudaDeviceSynchronize();

    // 2) Solver BWM
    double* gammas = (double*)malloc(sizeof(double));
    gammas[0] = gamma;
    bwm::BoundaryWallMethod solver(boundary, num_b, gammas, 1, std::sqrt(k2), angle);
    cudaDeviceSynchronize();

    // 3) Malla y volcado de density.dat y phase.dat
    start = std::chrono::high_resolution_clock::now();
    std::ofstream foutD("density.dat"), foutP("phase.dat");
    double xmin=-8, xmax=8;
    Point *P = (Point*)malloc(sizeof(Point));
    for(int i = 0; i < Ngrid; ++i){
        double y = xmin + i*(xmax - xmin) / (Ngrid - 1);
        for(int j = 0; j < Ngrid; ++j){
            double x = xmin + j*(xmax-xmin)/(Ngrid-1);
            Point p = {x,y};
            P[0] = p;
            //std::cout << "INICIO PUNTO (" << x << " , " << y << ")\n";
            auto psi = solver.computeScatteredWave(P, Ngrid)[0];
            cudaDeviceSynchronize();
            foutD << (psi.x * psi.x + psi.y * psi.y) << (j+1==Ngrid? "\n":" ");
            foutP << atan2(psi.y, psi.x) << (j+1==Ngrid? "\n":" ");
            //std::cout << "FIN PUNTO (" << x << " , " << y << ")\n";
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    std::cout << num_b << "\t" << elapsed.count() << "\n";

    return 0;
}