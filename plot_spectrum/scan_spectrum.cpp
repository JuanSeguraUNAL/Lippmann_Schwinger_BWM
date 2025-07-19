#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include "ConfocalParabolicBilliard.hpp"
#include "BoundaryWallMethod.hpp"

int main(int argc, char* argv[]){
    if(argc < 3){
        std::cerr << "Uso: " << argv[0] << " xi0 eta0\n";
        return 1;
    }

    double xi0  = std::stod(argv[1]);
    double eta0 = std::stod(argv[2]);

    // Parámetros fijos de escaneo
    double gamma          = std::numeric_limits<double>::infinity();
    double angle          = 0.0;
    double k_min          = 0.1;
    double k_max          = 2.0;
    int    num_k          = 100;
    double refine_window  = 0.05;
    int    refine_iters   = 2;
    double tolerance      = 1e-4;

    // 1) frontera y solver
    bwm::ConfocalParabolicBilliard billiard(xi0, eta0, 200);
    auto boundary = billiard.getBoundary();
    std::vector<double> gam{ gamma };
    bwm::BoundaryWallMethod solver(boundary, gam, 1.0, angle);

    // 2) escanear espectro
    auto spec = solver.scanSpectrum(
        k_min, k_max, num_k,
        refine_window,
        refine_iters,
        tolerance
    );

    // 3) volcar spectrum.dat (k, norm)
    std::ofstream ofs("spectrum.dat");
    for(size_t i=0; i<spec.k_values.size(); ++i)
        ofs << spec.k_values[i] << " " 
            << spec.norm_values[i] << "\n";
    ofs.close();

    // 4) volcar resonances.dat (valores de k)
    std::ofstream ofr("resonances.dat");
    for(double r: spec.resonances)
        ofr << r << "\n";
    ofr.close();

    return 0;
}
