/*
 * Copyright (c) The acados authors.
 * 
 * This file is part of acados.
 * 
 * The 2-Clause BSD License
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_car_model.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

#define NX     CAR_MODEL_NX
#define NZ     CAR_MODEL_NZ
#define NU     CAR_MODEL_NU
#define NP     CAR_MODEL_NP
#define NBX    CAR_MODEL_NBX
#define NBX0   CAR_MODEL_NBX0
#define NBU    CAR_MODEL_NBU
#define NSBX   CAR_MODEL_NSBX
#define NSBU   CAR_MODEL_NSBU
#define NSH    CAR_MODEL_NSH
#define NSG    CAR_MODEL_NSG
#define NSPHI  CAR_MODEL_NSPHI
#define NSHN   CAR_MODEL_NSHN
#define NSGN   CAR_MODEL_NSGN
#define NSPHIN CAR_MODEL_NSPHIN
#define NSBXN  CAR_MODEL_NSBXN
#define NS     CAR_MODEL_NS
#define NSN    CAR_MODEL_NSN
#define NG     CAR_MODEL_NG
#define NBXN   CAR_MODEL_NBXN
#define NGN    CAR_MODEL_NGN
#define NY0    CAR_MODEL_NY0
#define NY     CAR_MODEL_NY
#define NYN    CAR_MODEL_NYN
#define NH     CAR_MODEL_NH
#define NPHI   CAR_MODEL_NPHI
#define NHN    CAR_MODEL_NHN
#define NPHIN  CAR_MODEL_NPHIN
#define NR     CAR_MODEL_NR


int main()
{

    car_model_solver_capsule *acados_ocp_capsule = car_model_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = CAR_MODEL_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = car_model_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("car_model_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = car_model_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = car_model_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = car_model_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = car_model_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = car_model_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = car_model_acados_get_nlp_opts(acados_ocp_capsule);

    // initial condition
    int idxbx0[NBX0];
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;

    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = 0;
    ubx0[0] = 0;
    lbx0[1] = 0;
    ubx0[1] = 0;
    lbx0[2] = 0;
    ubx0[2] = 0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    // set parameters
    double p[NP];
    p[0] = 28;
    p[1] = 15;
    p[2] = 48;
    p[3] = 17;
    p[4] = 25;
    p[5] = 30;
    p[6] = 16;
    p[7] = 18;
    p[8] = 24;
    p[9] = 38;
    p[10] = 30;
    p[11] = 22;
    p[12] = 36;
    p[13] = 44;
    p[14] = 37;
    p[15] = 32;
    p[16] = 40;
    p[17] = 31;
    p[18] = 37;
    p[19] = 29;

    for (int ii = 0; ii <= N; ii++)
    {
        car_model_acados_update_params(acados_ocp_capsule, ii, p, NP);
    }
  

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];


    // solve ocp in loop
    int rti_phase = 0;

    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
        status = car_model_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat( NX, N+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat( NU, N, utraj, NU );
    // ocp_nlp_out_print(nlp_solver->dims, nlp_out);

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("car_model_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("car_model_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);

    car_model_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);

    // free solver
    status = car_model_acados_free(acados_ocp_capsule);
    if (status) {
        printf("car_model_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = car_model_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("car_model_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
