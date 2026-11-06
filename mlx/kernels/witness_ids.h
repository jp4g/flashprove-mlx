// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Names shared by the host and the witness kernels: this file is included by both.
#pragma once
enum WMat { WM_AF, WM_AD, WM_AR, WM_QA, WM_XI, WM_XO, WM_COUNT };
enum WTensor { WT_GI, WT_G0, WT_GC, WT_GH, WT_SL, WT_UP, WT_RG, WT_RU, WT_XS, WT_RO, WT_RD, WT_AW, WT_RA, WT_MW, WT_RM, WT_QKW, WT_RQK, WT_VW, WT_RV,
               WT_AN, WT_MN, WT_XR, WT_LR, WT_RL, WT_SEL, WT_SC, WT_ROW, WT_LY, WT_L3, WT_Q, WT_KX, WT_V, WT_O, WT_XBI, WT_XBO, WT_COUNT };
enum WSeg { WS_ROW, WS_N1, WS_N2, WS_N1H, WS_N2H, WS_CN1, WS_CN2, WS_QK, WS_QKH, WS_CQK, WS_V, WS_VH, WS_CV, WS_O, WS_D, WS_OH, WS_DH, WS_G, WS_U, WS_GH, WS_UH, WS_UT,
            WS_G0, WS_CG, WS_F, WS_SEL, WS_LT, WS_LH, WS_LR, WS_L3, WS_CLR, WS_DD, WS_MR, WS_XR, WS_COUNT };
// Words of a kernel's parameter array p.
enum WParam { WP_T, WP_D, WP_F, WP_FE, WP_NE, WP_NQK, WP_DQ, WP_DKV, WP_XL, WP_NR, WP_HQ, WP_HV, WP_HO, WP_HG, WP_HU, WP_HD, WP_HR, WP_GCL, WP_RCL, WP_LAM, WP_LAMM, WP_I, WP_EX,
              WP_KQ, WP_KK, WP_KV, WP_KO, WP_KD, WP_KG, WP_KU, WP_KGR, WP_KUR, WP_KR, WP_KN, WP_COUNT };
// One norm row in the words of the kernel's output.
enum { WN_A, WN_B, WN_SIG, WN_RLO, WN_RHI, WN_R, WN_D, WN_W = WN_D + 99, WN_STRIDE = WN_W + 9 };
// The attention segment's parameters.
enum AParam { AP_T, AP_BT, AP_NT, AP_HP, AP_H, AP_HA, AP_DH, AP_DQ, AP_DKV, AP_LAM, AP_LAMS, AP_G, AP_LT, AP_LDH, AP_NM, AP_ND, AP_NP, AP_LMS, AP_KPART, AP_SH, AP_C, AP_RHO, AP_OM, AP_NSEG,
              AP_MS, AP_AHK, AP_AHFR, AP_AOK, AP_AOFR, AP_OZR, AP_OIB, AP_ORHP, AP_U0, AP_COUNT };
enum { A_RM, A_RD, A_FM, A_FD, A_WL, A_WH, A_IX };
// The head's parameters.
enum HParam { HP_T, HP_D, HP_V, HP_JV, HP_JVA, HP_KF, HP_MF, HP_EPS, HP_SH, HP_F, HP_NRH, HP_HFK, HP_HFFR, HP_HFROWS, HP_HDK, HP_HDFR, HP_HRK, HP_HRFR, HP_XK, HP_XFR, HP_OD1, HP_OD2, HP_OE, HP_OFFE, HP_REACH, HP_COUNT };
enum HSeg { HS_NF, HS_NFH, HS_CNF, HS_HD0, HS_HD1, HS_HD2, HS_HE0, HS_HE1, HS_ROW, HS_COUNT };
