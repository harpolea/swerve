#ifndef UNIT_TESTS_H
#define UNIT_TESTS_H

void run_cuda_tests();
float r();

bool test_invert_matrix();
bool test_cons_to_prim_comp();
bool test_nan_check();
bool test_zbrent();
bool test_W_swe();
bool test_phi();
bool test_p_from_rho_eps();
bool test_phi_from_p();
bool test_f_of_p();
bool test_p_from_swe();
bool test_getNumKernels();
bool test_getNumBlocksAndThreads();

#endif
