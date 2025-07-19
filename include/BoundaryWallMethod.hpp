#ifndef BWM_BOUNDARY_WALL_METHOD_HPP
#define BWM_BOUNDARY_WALL_METHOD_HPP

#include <Eigen/Dense>
#include <boost/math/special_functions/hankel.hpp>
#include <complex>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace bwm {

class BoundaryWallMethod {
public:
    using Complex   = std::complex<double>;
    using Vector2d  = Eigen::Vector2d;
    using VectorXcd = Eigen::VectorXcd;
    using MatrixXcd = Eigen::MatrixXcd;

    struct Spectrum {
        std::vector<double> k_values;
        std::vector<double> norm_values;
        std::vector<double> resonances;
    };

    BoundaryWallMethod(const std::vector<Vector2d>& boundary_points,
                       const std::vector<double>& gamma_vals,
                       double k_, double angle_,
                       double alpha_ = 1.0)
      : boundary(boundary_points),
        N(boundary_points.size()),
        k(k_), angle(angle_), alpha(alpha_)
    {
        if (gamma_vals.size() == 1)
            gamma.assign(N, gamma_vals[0]);
        else
            gamma = gamma_vals;

        kx = k * std::cos(angle);
        ky = k * std::sin(angle);

        segment_lengths = calculateSegmentLengths();
        M_flat          = buildMMatrixFlat();
        T               = buildTMatrix();
    }

    Complex incidentWave(double x, double y) const {
        static const Complex I(0.0, 1.0);
        return std::exp(I * (kx*x + ky*y));
    }

    Spectrum scanSpectrum(double k_min=0.1, double k_max=2.0,
                          int num_points=100,
                          double refine_window=0.05,
                          int refine_iterations=2,
                          double tolerance=1e-4)
    {
        auto [ks, norms] = initialScan(k_min, k_max, num_points);
        auto peaks       = findPeaks(ks, norms, k_min, k_max);
        auto refined     = refineResonances(peaks,
                                            refine_window,
                                            refine_iterations,
                                            tolerance,
                                            num_points/10);
        return { std::move(ks), std::move(norms), std::move(refined) };
    }

    std::vector<Complex> computeScatteredWave(const std::vector<Vector2d>& obs) {
        // Compute the incident wave over the boundary
        VectorXcd phi(N);
        for (size_t i = 0; i < N; ++i)
            phi[i] = incidentWave(boundary[i].x(), boundary[i].y());

        // Multiply the interaction matrix T with the vector phi
        VectorXcd Tphi = T * phi;

        // Evaluate the total wave over every observation point
        std::vector<Complex> psi(obs.size());
        for (size_t i = 0; i < obs.size(); ++i) {
            psi[i] = incidentWave(obs[i].x(), obs[i].y());
            // Add the contribution of the scattered wave
            for (size_t j = 0; j < N; ++j) {
                Complex G = green(obs[i], boundary[j]);
                psi[i] += G * segment_lengths[j] * Tphi[j];
            }
        }
        return psi;
    }

private:
    // Index of a 2D matrix with row major
    size_t idx(size_t i, size_t j) const { return i*N + j; }
    // Given the index, calculate the (i,j) position in the matrix
    std::pair<size_t,size_t> ij(size_t id) const { return { id/N, id%N }; }

    // Compute the length of the segments of the boundary
    std::vector<double> calculateSegmentLengths() const {
        std::vector<double> L(N);                      // Vector L will store the segment lengths
        for (size_t i = 0; i < N; ++i) {
            size_t j = (i+1)%N;                        // The modulus operation %N closes the bounday
            L[i] = (boundary[j] - boundary[i]).norm(); // Substract the consecutive boundary positions and compute the euclidean norm
        }
        return L;
    }

    // Compute the Green function 
    Complex green(const Vector2d& r1, const Vector2d& r2) const {
        double R = (r1 - r2).norm();                                  // Distance between the observation point and the boundary point
        if (R < 1e-10) return handleDiagonal();                       // If the observation point is near the boundary point, avoid infinite values
        static const Complex I(0.0, 1.0);                             // Declare the complex unit
        return alpha*I*(1.0/4.0) * boost::math::cyl_hankel_1(0, k*R); // Compute the Green function given the Hankel function of order zero
    }

    // Treat the points near the boundary by approximating them to the Green function in a mean boundary point times the segment lenght
    Complex handleDiagonal() const {
        double avg = std::accumulate(segment_lengths.begin(),
                                     segment_lengths.end(), 0.0) / N;
        static const Complex I(0.0, 1.0);
        return alpha*I*(1.0/4.0)
             * boost::math::cyl_hankel_1(0, k*(avg/2.0))
             * avg;
    }

    // Build the boundary interaction matrix M
    std::vector<Complex> buildMMatrixFlat() {
        std::vector<Complex> M(N*N);
        for (size_t id = 0; id < N*N; ++id) {
            auto [i,j] = ij(id);
            // If the entry Mij is diagonal call handleDiagonal(), otherwise compute the Green funtion times the segment lenght
            M[id] = (i==j
                     ? handleDiagonal()
                     : green(boundary[i], boundary[j]) * segment_lengths[j]);
        }
        return M;
    }

    // Build the interaction matrix
    MatrixXcd buildTMatrix() {
        // Given the matrix in its flat form in row major, obtain the same matrix as a 2D matrix structure (for Eigen)
        MatrixXcd Mmat(N,N);
        for (size_t i=0;i<N;++i)
            for (size_t j=0;j<N;++j)
                Mmat(i,j) = M_flat[i*N+j];
        
        // Verify if all gamma values are infinite
        bool all_inf = std::all_of(gamma.begin(), gamma.end(),
                                   [](double g){ return std::isinf(g); });

        // If all values are infinite, return the negative of the inverse of M
        if (all_inf) {
            return -Mmat.inverse();
        } 
        // Otherwise, compute T = G * (I - G * M)^{-1}, where G is a diagonal matrix with the gamma values
        else {
            Eigen::VectorXd gvec = Eigen::Map<const Eigen::VectorXd>(gamma.data(), N);
            Eigen::MatrixXd G    = gvec.asDiagonal();
            MatrixXcd Iden = MatrixXcd::Identity(N,N);
            return G.cast<Complex>() * ((Iden - G.cast<Complex>()*Mmat).inverse());
        }
    }

    // Initial scan in order to find possible resonances given a wave number range
    std::pair<std::vector<double>,std::vector<double>>
    initialScan(double k_min, double k_max, int np) {
        // Initializes ks (stores the wave numbers) and norms (stores the norms of the matrix T evaluated in the wave numbers)
        std::vector<double> ks(np), norms(np);
        double dk = (k_max - k_min) / (np-1);
        for (int i=0;i<np;++i) {
            ks[i] = k_min + i*dk;
            updateK(ks[i]);
            norms[i] = T.array().abs().sum();  // This norm is the sum of the absolute values of every entrie of the matrix
        }
        return {ks, norms};
    }

    // Finds local peaks given the norms generated by initial scan
    std::vector<double> findPeaks(const std::vector<double>& ks,
                                  const std::vector<double>& norms,
                                  double k_min, double k_max) const
    {   
        // Initialize auxiliary vectors
        std::vector<bool> valid(norms.size(), true);  // Indicates if the points can be considered in the peak search
        std::vector<double> peaks;                    // Stores the values k in which there is a peak in norms

        // This while loop will break up only if there are no peaks left to detect
        while (true) {
            double mv = -std::numeric_limits<double>::infinity(); // Stablish norms biggest value to -infinity
            int midx = -1;                                        // Index of the biggest value
            // Find the biggest value in norms
            for (size_t i=0;i<norms.size();++i){
                if (valid[i] && norms[i]>mv){
                    mv = norms[i];
                    midx = i;
                }
            }
            if (midx<0) break; // If nothing could be found, break the while loop

            // Deactivate the peak and it's near values
            valid[midx] = false;
            for (int i=midx+1; i+1<(int)norms.size(); ++i) {
                if (!valid[i]) continue;
                if (norms[i]>norms[i-1]) break;
                valid[i] = false;
            }
            for (int i=midx-1; i>0; --i) {
                if (!valid[i]) continue;
                if (norms[i]>norms[i+1]) break;
                valid[i] = false;
            }

            // Store the peak if it isn't in the extremes of the k values
            double kp = ks[midx];
            if (kp!=k_min && kp!=k_max)
                peaks.push_back(kp);
        }
        return peaks;
    }

    // Refine with more precision the resonances found by findpeaks()
    std::vector<double> refineResonances(const std::vector<double>& peaks,
                                         double window, int iter,
                                         double tol, int ppw)
    {
        std::vector<double> refined;        // This vector stores the refined frecuencies
        // Iterate over every aproximate peak
        for (double kg : peaks) {
            // ck is the actual value which will be refined and w is the initial width of the local search interval around ck
            double ck = kg, w = window;
            // The refine process will be repeated iter times, each iteration reduces the width w in order to obtain better accuracy
            for (int it=0; it<iter; ++it) {
                // kws : vector with ppw (points per window) elements centered at ck and of width 2w
                std::vector<double> kws(ppw), vals(ppw);
                const double W = (2*w)/(ppw-1);          // Auxiliary value
                for (int j=0;j<ppw;++j) {
                    kws[j] = ck - w + j*W;
                    updateK(kws[j]);
                    vals[j] = T.array().abs().sum();     // Compute the norm as the sum of absolute values of every element of matrix T
                }
                // Choose the best local peak (k value which maximizes the "norm" of T inside the interval)
                auto im = std::max_element(vals.begin(), vals.end());
                double nk = kws[std::distance(vals.begin(), im)];     // New candidate to be the refined peak
                // Verify convergence
                if (std::abs(nk - ck) < tol) break;
                ck = nk; w *= 0.1;
            }
            refined.push_back(ck);  // Store the refined value
        }
        return refined;
    }

    // Updates the wave number and every computation which depends of it
    void updateK(double newk) {
        k  = newk;
        kx = k*std::cos(angle);
        ky = k*std::sin(angle);
        M_flat = buildMMatrixFlat();
        T      = buildTMatrix();
    }

    std::vector<Vector2d> boundary;
    size_t                N;
    std::vector<double>   gamma;
    double                k, angle, alpha, kx, ky;
    std::vector<double>   segment_lengths;
    std::vector<Complex>  M_flat;
    MatrixXcd             T;
};

} 

#endif 
