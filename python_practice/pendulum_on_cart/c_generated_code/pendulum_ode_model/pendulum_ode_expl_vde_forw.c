/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) pendulum_ode_expl_vde_forw_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[3] = {0, 0, 0};

/* pendulum_ode_expl_vde_forw:(i0[4],i1[4x4],i2[4],i3,i4[])->(o0[4],o1[4x4],o2[4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][2] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][3] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a1=-8.0000000000000016e-02;
  a2=arg[0]? arg[0][1] : 0;
  a3=sin(a2);
  a4=(a1*a3);
  a5=(a4*a0);
  a6=(a5*a0);
  a7=9.8100000000000009e-01;
  a8=cos(a2);
  a9=(a7*a8);
  a10=(a9*a3);
  a6=(a6+a10);
  a10=arg[3]? arg[3][0] : 0;
  a6=(a6+a10);
  a11=1.1000000000000001e+00;
  a12=1.0000000000000001e-01;
  a13=(a12*a8);
  a14=(a13*a8);
  a11=(a11-a14);
  a6=(a6/a11);
  if (res[0]!=0) res[0][2]=a6;
  a14=(a1*a8);
  a15=(a14*a3);
  a16=(a15*a0);
  a17=(a16*a0);
  a18=(a10*a8);
  a17=(a17+a18);
  a18=1.0791000000000002e+01;
  a19=(a18*a3);
  a17=(a17+a19);
  a19=8.0000000000000004e-01;
  a20=(a19*a11);
  a17=(a17/a20);
  if (res[0]!=0) res[0][3]=a17;
  a21=arg[1]? arg[1][2] : 0;
  if (res[1]!=0) res[1][0]=a21;
  a21=arg[1]? arg[1][3] : 0;
  if (res[1]!=0) res[1][1]=a21;
  a22=cos(a2);
  a23=arg[1]? arg[1][1] : 0;
  a24=(a22*a23);
  a25=(a1*a24);
  a25=(a0*a25);
  a26=(a4*a21);
  a25=(a25+a26);
  a25=(a0*a25);
  a26=(a5*a21);
  a25=(a25+a26);
  a26=(a9*a24);
  a27=sin(a2);
  a23=(a27*a23);
  a28=(a7*a23);
  a28=(a3*a28);
  a26=(a26-a28);
  a25=(a25+a26);
  a25=(a25/a11);
  a26=(a6/a11);
  a28=(a12*a23);
  a28=(a8*a28);
  a29=(a13*a23);
  a28=(a28+a29);
  a29=(a26*a28);
  a25=(a25-a29);
  if (res[1]!=0) res[1][2]=a25;
  a25=(a14*a24);
  a29=(a1*a23);
  a29=(a3*a29);
  a25=(a25-a29);
  a25=(a0*a25);
  a29=(a15*a21);
  a25=(a25+a29);
  a25=(a0*a25);
  a21=(a16*a21);
  a25=(a25+a21);
  a23=(a10*a23);
  a25=(a25-a23);
  a24=(a18*a24);
  a25=(a25+a24);
  a25=(a25/a20);
  a24=(a17/a20);
  a28=(a19*a28);
  a28=(a24*a28);
  a25=(a25-a28);
  if (res[1]!=0) res[1][3]=a25;
  a25=arg[1]? arg[1][6] : 0;
  if (res[1]!=0) res[1][4]=a25;
  a25=arg[1]? arg[1][7] : 0;
  if (res[1]!=0) res[1][5]=a25;
  a28=arg[1]? arg[1][5] : 0;
  a23=(a22*a28);
  a21=(a1*a23);
  a21=(a0*a21);
  a29=(a4*a25);
  a21=(a21+a29);
  a21=(a0*a21);
  a29=(a5*a25);
  a21=(a21+a29);
  a29=(a9*a23);
  a28=(a27*a28);
  a30=(a7*a28);
  a30=(a3*a30);
  a29=(a29-a30);
  a21=(a21+a29);
  a21=(a21/a11);
  a29=(a12*a28);
  a29=(a8*a29);
  a30=(a13*a28);
  a29=(a29+a30);
  a30=(a26*a29);
  a21=(a21-a30);
  if (res[1]!=0) res[1][6]=a21;
  a21=(a14*a23);
  a30=(a1*a28);
  a30=(a3*a30);
  a21=(a21-a30);
  a21=(a0*a21);
  a30=(a15*a25);
  a21=(a21+a30);
  a21=(a0*a21);
  a25=(a16*a25);
  a21=(a21+a25);
  a28=(a10*a28);
  a21=(a21-a28);
  a23=(a18*a23);
  a21=(a21+a23);
  a21=(a21/a20);
  a29=(a19*a29);
  a29=(a24*a29);
  a21=(a21-a29);
  if (res[1]!=0) res[1][7]=a21;
  a21=arg[1]? arg[1][10] : 0;
  if (res[1]!=0) res[1][8]=a21;
  a21=arg[1]? arg[1][11] : 0;
  if (res[1]!=0) res[1][9]=a21;
  a29=arg[1]? arg[1][9] : 0;
  a23=(a22*a29);
  a28=(a1*a23);
  a28=(a0*a28);
  a25=(a4*a21);
  a28=(a28+a25);
  a28=(a0*a28);
  a25=(a5*a21);
  a28=(a28+a25);
  a25=(a9*a23);
  a29=(a27*a29);
  a30=(a7*a29);
  a30=(a3*a30);
  a25=(a25-a30);
  a28=(a28+a25);
  a28=(a28/a11);
  a25=(a12*a29);
  a25=(a8*a25);
  a30=(a13*a29);
  a25=(a25+a30);
  a30=(a26*a25);
  a28=(a28-a30);
  if (res[1]!=0) res[1][10]=a28;
  a28=(a14*a23);
  a30=(a1*a29);
  a30=(a3*a30);
  a28=(a28-a30);
  a28=(a0*a28);
  a30=(a15*a21);
  a28=(a28+a30);
  a28=(a0*a28);
  a21=(a16*a21);
  a28=(a28+a21);
  a29=(a10*a29);
  a28=(a28-a29);
  a23=(a18*a23);
  a28=(a28+a23);
  a28=(a28/a20);
  a25=(a19*a25);
  a25=(a24*a25);
  a28=(a28-a25);
  if (res[1]!=0) res[1][11]=a28;
  a28=arg[1]? arg[1][14] : 0;
  if (res[1]!=0) res[1][12]=a28;
  a28=arg[1]? arg[1][15] : 0;
  if (res[1]!=0) res[1][13]=a28;
  a25=arg[1]? arg[1][13] : 0;
  a22=(a22*a25);
  a23=(a1*a22);
  a23=(a0*a23);
  a29=(a4*a28);
  a23=(a23+a29);
  a23=(a0*a23);
  a29=(a5*a28);
  a23=(a23+a29);
  a29=(a9*a22);
  a27=(a27*a25);
  a25=(a7*a27);
  a25=(a3*a25);
  a29=(a29-a25);
  a23=(a23+a29);
  a23=(a23/a11);
  a29=(a12*a27);
  a29=(a8*a29);
  a25=(a13*a27);
  a29=(a29+a25);
  a26=(a26*a29);
  a23=(a23-a26);
  if (res[1]!=0) res[1][14]=a23;
  a23=(a14*a22);
  a26=(a1*a27);
  a26=(a3*a26);
  a23=(a23-a26);
  a23=(a0*a23);
  a26=(a15*a28);
  a23=(a23+a26);
  a23=(a0*a23);
  a28=(a16*a28);
  a23=(a23+a28);
  a27=(a10*a27);
  a23=(a23-a27);
  a22=(a18*a22);
  a23=(a23+a22);
  a23=(a23/a20);
  a29=(a19*a29);
  a24=(a24*a29);
  a23=(a23-a24);
  if (res[1]!=0) res[1][15]=a23;
  a23=arg[2]? arg[2][2] : 0;
  if (res[2]!=0) res[2][0]=a23;
  a23=arg[2]? arg[2][3] : 0;
  if (res[2]!=0) res[2][1]=a23;
  a24=(1./a11);
  a29=cos(a2);
  a22=arg[2]? arg[2][1] : 0;
  a29=(a29*a22);
  a27=(a1*a29);
  a27=(a0*a27);
  a4=(a4*a23);
  a27=(a27+a4);
  a27=(a0*a27);
  a5=(a5*a23);
  a27=(a27+a5);
  a9=(a9*a29);
  a2=sin(a2);
  a2=(a2*a22);
  a7=(a7*a2);
  a7=(a3*a7);
  a9=(a9-a7);
  a27=(a27+a9);
  a27=(a27/a11);
  a6=(a6/a11);
  a12=(a12*a2);
  a12=(a8*a12);
  a13=(a13*a2);
  a12=(a12+a13);
  a6=(a6*a12);
  a27=(a27-a6);
  a24=(a24+a27);
  if (res[2]!=0) res[2][2]=a24;
  a8=(a8/a20);
  a14=(a14*a29);
  a1=(a1*a2);
  a3=(a3*a1);
  a14=(a14-a3);
  a14=(a0*a14);
  a15=(a15*a23);
  a14=(a14+a15);
  a0=(a0*a14);
  a16=(a16*a23);
  a0=(a0+a16);
  a10=(a10*a2);
  a0=(a0-a10);
  a18=(a18*a29);
  a0=(a0+a18);
  a0=(a0/a20);
  a17=(a17/a20);
  a19=(a19*a12);
  a17=(a17*a19);
  a0=(a0-a17);
  a8=(a8+a0);
  if (res[2]!=0) res[2][3]=a8;
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real pendulum_ode_expl_vde_forw_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_expl_vde_forw_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_expl_vde_forw_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    case 3: return casadi_s2;
    case 4: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif