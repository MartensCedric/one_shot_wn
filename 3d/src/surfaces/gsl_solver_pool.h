#pragma once

#include <gsl/gsl_multiroots.h>

namespace gwn3d {

// RAII wrapper around a 3-equation GSL Hybrids root solver. The default
// constructor allocates; the destructor frees the solver (which also frees
// the bound x vector).
struct GslHybridsSolver3 {
    gsl_multiroot_fsolver* solver = nullptr;
    gsl_multiroot_function F{};

    explicit GslHybridsSolver3(int (*f)(const gsl_vector*, void*, gsl_vector*)) {
        F.f = f;
        F.n = 3;
        solver = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrids, F.n);
        solver->x = gsl_vector_alloc(F.n);
        solver->function = &F;
    }

    ~GslHybridsSolver3() {
        if (solver) gsl_multiroot_fsolver_free(solver);
    }

    GslHybridsSolver3(const GslHybridsSolver3&) = delete;
    GslHybridsSolver3& operator=(const GslHybridsSolver3&) = delete;
};

} // namespace gwn3d
