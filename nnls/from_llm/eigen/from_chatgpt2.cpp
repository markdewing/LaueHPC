
#define EIGEN_USE_MKL_ALL
// Fed nnls_from_chatgpt.py to ChatGPT and ask for C++ version using Eigen
#include <iostream>
#include <Eigen/Dense>
#include <set>

using namespace Eigen;

VectorXd non_negative_least_squares(const MatrixXd& A, const VectorXd& y, double epsilon = 1e-6) {
    int m = A.rows();
    int n = A.cols();
    VectorXd x = VectorXd::Zero(n);  // Initialize the solution vector x with zeros
    std::set<int> P;  // Initialize the set P to store selected indices
    std::set<int> R;  // Initialize the set R with all indices
    for (int i = 0; i < n; ++i) {
        R.insert(i);
    }

    VectorXd w = A.transpose() * (y - A * x);
    VectorXd wr(R.size());

    //for (int i = 0; i < n; ++i) {
    //    wr(i) = w(R[i]);
    //}
    int idx = 0;
    for (auto ri = R.begin(); ri != R.end(); ri++) 
    {
        wr[idx] = w[*ri];
        idx++;
    }

    // Create storage for AP
    MatrixXd AP = A;

    
    // Create storage for s
    VectorXd s = VectorXd::Zero(n);  // Initialize a vector s with zeros

    while (!R.empty() && wr.maxCoeff() > epsilon) {
        double max_dot_product = -std::numeric_limits<double>::infinity();
        int j_max = -1;

        for (int j : R) {
            double dot_product = w(j);
            if (dot_product > max_dot_product) {
                max_dot_product = dot_product;
                j_max = j;
            }
        }

        P.insert(j_max);  // Add the selected index to P
        R.erase(j_max);   // Remove the selected index from R

        //AP = AP.colwise().take(P);
        std::vector<int> pidx(P.begin(), P.end());
        //std::cout << " pidx size = " << pidx.size() << std::endl;
        // Construct the submatrix AP consisting of columns corresponding to indices in P
        AP = A(Eigen::placeholders::all, pidx);

          
        VectorXd sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);  // Compute the least squares solution for the selected indices

        //for (int i = 0; i < P.size(); ++i) {
        //    s(P[i]) = sP(i);
        //}
        s = VectorXd::Zero(n);  // Initialize a vector s with zeros
        int idx = 0;
        for (auto pi = P.begin(); pi != P.end(); pi++) 
        {
            s[*pi] = sP(idx);
            idx++;
        }

        while (sP.minCoeff() <= 0) {
            double alpha = std::numeric_limits<double>::infinity();  // Initialize alpha as positive infinity

            for (int i : P) {
                if (s(i) <= 0) {
                    double alpha_candidate = x(i) / (x(i) - s(i));
                    if (alpha_candidate < alpha) {
                        alpha = alpha_candidate;
                    }
                }
            }

            // Update the solution vector x with the computed alpha
            x += alpha * (s - x);

            // Move indices from P to R if their corresponding elements in x become non-positive
            for (auto it = P.begin(); it != P.end();) {
                int i = *it;
                if (x(i) <= 0.0) {
                    it = P.erase(it);
                    R.insert(i);
                } else {
                    ++it;
                }
            }

            // Recompute the submatrix AP and the least squares solution sP
            //AP = A;
            //AP = AP.colwise().take(P);
            std::vector<int> pidx(P.begin(), P.end());
            //AP = AP(pidx);
            AP = A(Eigen::placeholders::all, pidx);
            //s = VectorXd::Zero(n);
            s = VectorXd::Zero(n);  // Initialize a vector s with zeros
            sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);

            //for (int i = 0; i < P.size(); ++i) {
            //    s(P[i]) = sP(i);
            //}
            int idx = 0;
            for (auto pi = P.begin(); pi != P.end(); pi++) 
            {
                s[*pi] = sP(idx);
                idx++;
            }
        }

        x = s;  // Update the solution vector x with the non-negative least squares solution
        w = A.transpose() * (y - A * x);
        wr.resize(R.size());

        idx = 0;
        for (int i : R) {
            wr(idx) = w(i);
            ++idx;
        }
    }

    return x;
}

VectorXf non_negative_least_squares_float(const MatrixXf& A, const VectorXf& y, double epsilon = 1e-6) {
    int m = A.rows();
    int n = A.cols();
    VectorXf x = VectorXf::Zero(n);  // Initialize the solution vector x with zeros
    std::set<int> P;  // Initialize the set P to store selected indices
    std::set<int> R;  // Initialize the set R with all indices
    for (int i = 0; i < n; ++i) {
        R.insert(i);
    }

    VectorXf w = A.transpose() * (y - A * x);
    VectorXf wr(R.size());

    //for (int i = 0; i < n; ++i) {
    //    wr(i) = w(R[i]);
    //}
    int idx = 0;
    for (auto ri = R.begin(); ri != R.end(); ri++) 
    {
        wr[idx] = w[*ri];
        idx++;
    }

    while (!R.empty() && wr.maxCoeff() > epsilon) {
        double max_dot_product = -std::numeric_limits<double>::infinity();
        int j_max = -1;

        for (int j : R) {
            double dot_product = w(j);
            if (dot_product > max_dot_product) {
                max_dot_product = dot_product;
                j_max = j;
            }
        }

        P.insert(j_max);  // Add the selected index to P
        R.erase(j_max);   // Remove the selected index from R

        // Construct the submatrix AP consisting of columns corresponding to indices in P
        MatrixXf AP = A;
        //AP = AP.colwise().take(P);
        std::vector<int> pidx(P.begin(), P.end());
        //std::cout << " pidx size = " << pidx.size() << std::endl;
        AP = A(Eigen::placeholders::all, pidx);

        VectorXf s = VectorXf::Zero(n);  // Initialize a vector s with zeros
        VectorXf sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);  // Compute the least squares solution for the selected indices

        //for (int i = 0; i < P.size(); ++i) {
        //    s(P[i]) = sP(i);
        //}
        int idx = 0;
        for (auto pi = P.begin(); pi != P.end(); pi++) 
        {
            s[*pi] = sP(idx);
            idx++;
        }

        while (sP.minCoeff() <= 0) {
            double alpha = std::numeric_limits<double>::infinity();  // Initialize alpha as positive infinity

            for (int i : P) {
                if (s(i) <= 0) {
                    double alpha_candidate = x(i) / (x(i) - s(i));
                    if (alpha_candidate < alpha) {
                        alpha = alpha_candidate;
                    }
                }
            }

            // Update the solution vector x with the computed alpha
            x += alpha * (s - x);

            // Move indices from P to R if their corresponding elements in x become non-positive
            for (auto it = P.begin(); it != P.end();) {
                int i = *it;
                if (x(i) <= 0.0) {
                    it = P.erase(it);
                    R.insert(i);
                } else {
                    ++it;
                }
            }

            // Recompute the submatrix AP and the least squares solution sP
            AP = A;
            //AP = AP.colwise().take(P);
            std::vector<int> pidx(P.begin(), P.end());
            //AP = AP(pidx);
            AP = A(Eigen::placeholders::all, pidx);
            s = VectorXf::Zero(n);
            sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);

            //for (int i = 0; i < P.size(); ++i) {
            //    s(P[i]) = sP(i);
            //}
            int idx = 0;
            for (auto pi = P.begin(); pi != P.end(); pi++) 
            {
                s[*pi] = sP(idx);
                idx++;
            }
        }

        x = s;  // Update the solution vector x with the non-negative least squares solution
        w = A.transpose() * (y - A * x);
        wr.resize(R.size());

        idx = 0;
        for (int i : R) {
            wr(idx) = w(i);
            ++idx;
        }
    }

    return x;
}

int main() {
    // Example usage:
    MatrixXd A(3, 2);
    A << 1, 2,
         3, 4,
         5, 6;

    VectorXd y(3);
    y << 7 , 8 , 9;
    double eps = 1e-6;
    auto x = non_negative_least_squares(A, y, eps);
    std::cout << x << std::endl;

}
