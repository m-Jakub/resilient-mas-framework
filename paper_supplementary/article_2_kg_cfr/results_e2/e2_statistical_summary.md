## SCQA8 multiscenario E2 statistical summary

### Input folders (run_ids)
- analysis/outputs/e2/exp_e2_20260313_223929
- analysis/outputs/e2/exp_e2_20260314_015807
- analysis/outputs/e2/exp_e2_20260314_052549

Combined CSV used: logs/corrected_e2_batch_20260314_multiscenario/E2_Turns_judge_fixed.csv

### e2_scqa8_results.txt (full)
```text

========================================================================
  DATA LOADING & PREPARATION
========================================================================
  Loaded: 2970 rows | 270 debates | 3 conditions | 3 scenario(s)
  Conditions : ['cfr_no_kg', 'kg_cfr_full', 'no_cfr_baseline']
  Scenario(s): ['cerberus_biocontainment', 'aegis_blackout', 'synapse_orbital_strike']
  Turn range : 1-11
  Shocks     : {nan: 2430, 'S4': 153, 'S2': 141, 'S1': 123, 'S3': 123}
  turn_c = turn_number - 6.000 (range -5.0 to 5.0)
  Judge-complete rows: 2968
  Shock-present rows : 540 ({'S4': 153, 'S2': 141, 'S1': 123, 'S3': 123})

========================================================================
  DESCRIPTIVE STATISTICS BY CONDITION
========================================================================

  -- clarity --
                   N    mean      sd  p25   med   p75
condition                                            
cfr_no_kg        989  0.7244  0.2812  0.8  0.85  0.90
kg_cfr_full      990  0.8376  0.0547  0.8  0.85  0.85
no_cfr_baseline  989  0.7373  0.2744  0.8  0.85  0.90

  -- cogency --
                   N    mean      sd  p25   med   p75
condition                                            
cfr_no_kg        989  0.6240  0.2845  0.6  0.75  0.80
kg_cfr_full      990  0.7463  0.0678  0.7  0.75  0.78
no_cfr_baseline  989  0.6239  0.2853  0.6  0.75  0.80

  -- relevance --
                   N    mean      sd   p25  med   p75
condition                                            
cfr_no_kg        989  0.7582  0.3100  0.85  0.9  0.92
kg_cfr_full      990  0.9062  0.0390  0.90  0.9  0.92
no_cfr_baseline  989  0.7648  0.3096  0.90  0.9  0.95

  -- overall --
                   N    mean      sd  p25   med   p75
condition                                            
cfr_no_kg        989  0.6881  0.2918  0.7  0.82  0.85
kg_cfr_full      990  0.8219  0.0577  0.8  0.83  0.85
no_cfr_baseline  989  0.6942  0.2906  0.7  0.83  0.88

  -- Mean judge scores at shock turns by condition --
                            clarity  cogency  relevance
condition       shock_turn                             
cfr_no_kg       S1            0.632    0.502      0.630
                S2            0.450    0.308      0.434
                S3            0.591    0.488      0.597
                S4            0.610    0.492      0.621
kg_cfr_full     S1            0.806    0.703      0.890
                S2            0.834    0.739      0.909
                S3            0.835    0.729      0.908
                S4            0.815    0.714      0.890
no_cfr_baseline S1            0.410    0.228      0.331
                S2            0.660    0.543      0.658
                S3            0.608    0.494      0.591
                S4            0.607    0.494      0.624

========================================================================
  STAGE 1 -- LMMs: clarity / cogency / relevance
========================================================================

  Model (SCQA8_v8 Â§E0):
    Score ~ C(condition, Treatment('no_cfr_baseline'))
          + C(shock_turn, Treatment('none'))
          + turn_c
          + C(condition, Treatment('no_cfr_baseline')):turn_c
          + (1 + turn_c | debate_id)
          + (1 | scenario_id)   [attempted; degenerate if n_scenarios=1]

  Main test: Condition:turn_c (differential slope over time).
  Reference condition = no_cfr_baseline.
  Reference shock     = none.


  -- DV = CLARITY --
    debates=270, scenarios=3
    [clarity] random-slope did not converge; trying intercept-only
    [clarity] intercept-only converged=False

                                                                             Coef. Std.Err.        z  P>|z|  [0.025  0.975]
Intercept                                                                    0.747    0.006  122.780  0.000   0.735   0.759
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]           -0.013    0.008   -1.600  0.110  -0.029   0.003
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]          0.100    0.008   12.134  0.000   0.084   0.117
C(shock_turn, Treatment(reference='none'))[T.S1]                            -0.085    0.017   -4.937  0.000  -0.119  -0.051
C(shock_turn, Treatment(reference='none'))[T.S2]                            -0.057    0.016   -3.534  0.000  -0.089  -0.025
C(shock_turn, Treatment(reference='none'))[T.S3]                            -0.040    0.017   -2.311  0.021  -0.074  -0.006
C(shock_turn, Treatment(reference='none'))[T.S4]                            -0.040    0.015   -2.552  0.011  -0.070  -0.009
turn_c                                                                       0.049    0.002   26.164  0.000   0.046   0.053
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c     0.002    0.003    0.764  0.445  -0.003   0.007
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c  -0.052    0.003  -19.727  0.000  -0.057  -0.047
Group Var                                                                    0.000                                         
  converged=False  |  random-effects type: intercept_only
  log-likelihood=776.133

  -- DV = COGENCY --
    debates=270, scenarios=3
    [cogency] random-slope did not converge; trying intercept-only
    [cogency] intercept-only converged=False

                                                                             Coef. Std.Err.        z  P>|z|  [0.025  0.975]
Intercept                                                                    0.635    0.006  110.207  0.000   0.624   0.646
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]           -0.000    0.008   -0.048  0.961  -0.016   0.015
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]          0.123    0.008   15.667  0.000   0.107   0.138
C(shock_turn, Treatment(reference='none'))[T.S1]                            -0.109    0.016   -6.661  0.000  -0.141  -0.077
C(shock_turn, Treatment(reference='none'))[T.S2]                            -0.063    0.015   -4.154  0.000  -0.093  -0.033
C(shock_turn, Treatment(reference='none'))[T.S3]                            -0.035    0.016   -2.131  0.033  -0.067  -0.003
C(shock_turn, Treatment(reference='none'))[T.S4]                            -0.039    0.015   -2.683  0.007  -0.068  -0.011
turn_c                                                                       0.060    0.002   33.803  0.000   0.057   0.064
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c    -0.003    0.002   -1.325  0.185  -0.008   0.002
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c  -0.063    0.002  -25.315  0.000  -0.068  -0.058
Group Var                                                                    0.000                                         
  converged=False  |  random-effects type: intercept_only
  log-likelihood=937.583

  -- DV = RELEVANCE --
    debates=270, scenarios=3
    [relevance] full spec failed (Singular matrix)
    [relevance] random-slope converged (no scenario RE)

                                                                             Coef. Std.Err.        z  P>|z|  [0.025  0.975]
Intercept                                                                    0.777    0.007  117.526  0.000   0.764   0.790
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]           -0.007    0.009   -0.779  0.436  -0.025   0.011
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]          0.142    0.009   15.704  0.000   0.124   0.159
C(shock_turn, Treatment(reference='none'))[T.S1]                            -0.111    0.018   -6.101  0.000  -0.147  -0.075
C(shock_turn, Treatment(reference='none'))[T.S2]                            -0.070    0.017   -4.123  0.000  -0.103  -0.037
C(shock_turn, Treatment(reference='none'))[T.S3]                            -0.052    0.018   -2.876  0.004  -0.088  -0.017
C(shock_turn, Treatment(reference='none'))[T.S4]                            -0.037    0.016   -2.264  0.024  -0.069  -0.005
turn_c                                                                       0.060    0.002   30.012  0.000   0.056   0.064
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c     0.000    0.003    0.119  0.906  -0.005   0.006
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c  -0.062    0.003  -22.112  0.000  -0.067  -0.056
Group Var                                                                    0.000                                         
Group x turn_c Cov                                                          -0.000                                         
turn_c Var                                                                   0.000                                         
  converged=True  |  random-effects type: slope_only
  log-likelihood=612.045

========================================================================
  STAGE 2 -- Dproc Reconstruction & LMM
========================================================================

  Dproc(t) = alpha * SR(t) - beta * DIS(t)

  Per SCQA8_v8 Â§Weights: 'parameters alpha and beta are calibrated on a
  development split in order to z-normalize contributions of SR and DIS.'

  *** IMPLEMENTATION NOTE ***
  compute_v5_metrics.py (the versioned metric source) computes SR and DIS
  as separate scores and does NOT include a Dproc composition function with
  fixed calibrated alpha/beta values.  No confirmed alpha/beta constants were
  found in the codebase.  This reconstruction therefore applies z-normalization
  to equate the units of SR and DIS (making both mean-0, sd-1 within this
  dataset) and sums them with equal weights as a proxy pending confirmed
  calibration.  Treat the resulting index as an equal-weight z-normalized
  proxy, NOT a verified implementation of the published calibrated weights.

  Columns: sr_v5 (available turns 4-11), dis_v5 (turns 2-11)
  Dproc requires BOTH -> eligible rows are turns 4-11 (720 / 90 debates).

  Dproc-eligible rows: 2160 (turns 4-11)
  sr_v5  : mean=0.1149  sd=0.0464
  dis_v5 : mean=0.0534  sd=0.1825
  Dproc  : mean=-0.0000  sd=1.3793  range=[-6.977, 6.804]

  -- Dproc by condition --
                 count    mean     std     min     25%     50%     75%     max
condition                                                                     
cfr_no_kg        720.0 -0.1872  1.5245 -6.9770 -1.3502  0.4944  1.0042  6.8037
kg_cfr_full      720.0  0.3659  0.9849 -5.7432  0.3015  0.5973  0.8249  1.4245
no_cfr_baseline  720.0 -0.1787  1.4890 -6.7603 -1.3085  0.5103  1.0236  1.8477

  -- Shock types in Dproc subset (turns 4+) --
shock_turn
none    1890
S4        96
S3        60
S2        57
S1        57

  -- DV = DPROC --
    debates=270, scenarios=3
    [dproc] random-slope did not converge; trying intercept-only
    [dproc] intercept-only converged=False

                                                                             Coef. Std.Err.        z  P>|z|  [0.025  0.975]
Intercept                                                                   -0.734    0.055  -13.455  0.000  -0.841  -0.627
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]           -0.049    0.073   -0.670  0.503  -0.193   0.095
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]          1.090    0.074   14.756  0.000   0.945   1.235
C(shock_turn, Treatment(reference='none'))[T.S1]                            -0.181    0.152   -1.189  0.235  -0.479   0.117
C(shock_turn, Treatment(reference='none'))[T.S2]                            -0.169    0.152   -1.111  0.266  -0.467   0.129
C(shock_turn, Treatment(reference='none'))[T.S3]                            -0.287    0.148   -1.936  0.053  -0.577   0.003
C(shock_turn, Treatment(reference='none'))[T.S4]                            -0.289    0.119   -2.429  0.015  -0.523  -0.056
turn_c                                                                       0.391    0.018   21.246  0.000   0.354   0.427
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c     0.028    0.025    1.098  0.272  -0.022   0.077
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c  -0.364    0.025  -14.429  0.000  -0.414  -0.315
Group Var                                                                    0.028    0.013                                
  converged=False  |  random-effects type: intercept_only
  log-likelihood=-3286.668

========================================================================
  MAIN TEST SUMMARY: Condition x TurnIndex Interaction
========================================================================
  SCQA8_v8 Â§E0: 'report Condition:TurnIndex as the main test of differential degradation over time'

  clarity    C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c
             beta=+0.00200  SE=0.00262  z=+0.764  p_raw=0.4449 (ns)
  clarity    C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c
             beta=-0.05164  SE=0.00262  z=-19.727  p_raw=0.0000 (***)
  cogency    C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c
             beta=-0.00328  SE=0.00248  z=-1.325  p_raw=0.1853 (ns)
  cogency    C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c
             beta=-0.06275  SE=0.00248  z=-25.315  p_raw=0.0000 (***)
  relevance  C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c
             beta=+0.00033  SE=0.00279  z=+0.119  p_raw=0.9056 (ns)
  relevance  C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c
             beta=-0.06158  SE=0.00278  z=-22.112  p_raw=0.0000 (***)
  dproc      C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c
             beta=+0.02761  SE=0.02514  z=+1.098  p_raw=0.2721 (ns)
  dproc      C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c
             beta=-0.36415  SE=0.02524  z=-14.429  p_raw=0.0000 (***)

========================================================================
  FDR CORRECTION (Benjamini-Hochberg)
========================================================================
  Applied across all non-intercept fixed effects in clarity, cogency, relevance, dproc (4 models).
  Total tests: 36   Rejected at FDR < 0.05: 25

  FDR < 0.10 (26 terms):
       dv                                                                       coef         p_raw         p_fdr  sig_fdr
  cogency                                                                     turn_c 1.780829e-250 6.410983e-249     True
relevance                                                                     turn_c 6.831147e-198 1.229606e-196     True
  clarity                                                                     turn_c 6.753460e-151 8.104152e-150     True
  cogency C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c 2.195450e-141 1.975905e-140     True
relevance C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c 2.408952e-108 1.734446e-107     True
    dproc                                                                     turn_c 3.601879e-100  2.161127e-99     True
  clarity C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c  1.255086e-86  6.454729e-86     True
relevance        C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]  1.428743e-55  6.429343e-55     True
  cogency        C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]  2.529412e-55  1.011765e-54     True
    dproc        C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]  2.831085e-49  1.019191e-48     True
    dproc C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c  3.384519e-47  1.107661e-46     True
  clarity        C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]  6.936514e-34  2.080954e-33     True
  cogency                           C(shock_turn, Treatment(reference='none'))[T.S1]  2.713447e-11  7.514162e-11     True
relevance                           C(shock_turn, Treatment(reference='none'))[T.S1]  1.057139e-09  2.718356e-09     True
  clarity                           C(shock_turn, Treatment(reference='none'))[T.S1]  7.953552e-07  1.908853e-06     True
  cogency                           C(shock_turn, Treatment(reference='none'))[T.S2]  3.262892e-05  7.341508e-05     True
relevance                           C(shock_turn, Treatment(reference='none'))[T.S2]  3.739445e-05  7.918825e-05     True
  clarity                           C(shock_turn, Treatment(reference='none'))[T.S2]  4.096337e-04  8.192675e-04     True
relevance                           C(shock_turn, Treatment(reference='none'))[T.S3]  4.021567e-03  7.619811e-03     True
  cogency                           C(shock_turn, Treatment(reference='none'))[T.S4]  7.295815e-03  1.313247e-02     True
  clarity                           C(shock_turn, Treatment(reference='none'))[T.S4]  1.070016e-02  1.834313e-02     True
    dproc                           C(shock_turn, Treatment(reference='none'))[T.S4]  1.515201e-02  2.479420e-02     True
  clarity                           C(shock_turn, Treatment(reference='none'))[T.S3]  2.084821e-02  3.263198e-02     True
relevance                           C(shock_turn, Treatment(reference='none'))[T.S4]  2.359314e-02  3.538971e-02     True
  cogency                           C(shock_turn, Treatment(reference='none'))[T.S3]  3.306603e-02  4.761508e-02     True
    dproc                           C(shock_turn, Treatment(reference='none'))[T.S3]  5.282221e-02  7.313845e-02    False

  Full FDR table (sorted by p_fdr):
       dv                                                                       coef         p_raw         p_fdr  sig_fdr
  cogency                                                                     turn_c 1.780829e-250 6.410983e-249     True
relevance                                                                     turn_c 6.831147e-198 1.229606e-196     True
  clarity                                                                     turn_c 6.753460e-151 8.104152e-150     True
  cogency C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c 2.195450e-141 1.975905e-140     True
relevance C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c 2.408952e-108 1.734446e-107     True
    dproc                                                                     turn_c 3.601879e-100  2.161127e-99     True
  clarity C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c  1.255086e-86  6.454729e-86     True
relevance        C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]  1.428743e-55  6.429343e-55     True
  cogency        C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]  2.529412e-55  1.011765e-54     True
    dproc        C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]  2.831085e-49  1.019191e-48     True
    dproc C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c  3.384519e-47  1.107661e-46     True
  clarity        C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]  6.936514e-34  2.080954e-33     True
  cogency                           C(shock_turn, Treatment(reference='none'))[T.S1]  2.713447e-11  7.514162e-11     True
relevance                           C(shock_turn, Treatment(reference='none'))[T.S1]  1.057139e-09  2.718356e-09     True
  clarity                           C(shock_turn, Treatment(reference='none'))[T.S1]  7.953552e-07  1.908853e-06     True
  cogency                           C(shock_turn, Treatment(reference='none'))[T.S2]  3.262892e-05  7.341508e-05     True
relevance                           C(shock_turn, Treatment(reference='none'))[T.S2]  3.739445e-05  7.918825e-05     True
  clarity                           C(shock_turn, Treatment(reference='none'))[T.S2]  4.096337e-04  8.192675e-04     True
relevance                           C(shock_turn, Treatment(reference='none'))[T.S3]  4.021567e-03  7.619811e-03     True
  cogency                           C(shock_turn, Treatment(reference='none'))[T.S4]  7.295815e-03  1.313247e-02     True
  clarity                           C(shock_turn, Treatment(reference='none'))[T.S4]  1.070016e-02  1.834313e-02     True
    dproc                           C(shock_turn, Treatment(reference='none'))[T.S4]  1.515201e-02  2.479420e-02     True
  clarity                           C(shock_turn, Treatment(reference='none'))[T.S3]  2.084821e-02  3.263198e-02     True
relevance                           C(shock_turn, Treatment(reference='none'))[T.S4]  2.359314e-02  3.538971e-02     True
  cogency                           C(shock_turn, Treatment(reference='none'))[T.S3]  3.306603e-02  4.761508e-02     True
    dproc                           C(shock_turn, Treatment(reference='none'))[T.S3]  5.282221e-02  7.313845e-02    False
  clarity          C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]  1.096078e-01  1.461437e-01    False
  cogency   C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c  1.852796e-01  2.382167e-01    False
    dproc                           C(shock_turn, Treatment(reference='none'))[T.S1]  2.346213e-01  2.912540e-01    False
    dproc   C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c  2.721178e-01  3.160077e-01    False
    dproc                           C(shock_turn, Treatment(reference='none'))[T.S2]  2.663611e-01  3.160077e-01    False
relevance          C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]  4.359957e-01  4.853172e-01    False
  clarity   C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c  4.448741e-01  4.853172e-01    False
    dproc          C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]  5.029704e-01  5.325569e-01    False
relevance   C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c  9.056063e-01  9.314808e-01    False
  cogency          C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]  9.614377e-01  9.614377e-01    False

  -- Condition x TurnIndex â€” raw vs FDR-corrected p --
  clarity    p_raw=0.4449  p_fdr=0.4853  coef=C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c
  clarity    p_raw=0.0000  p_fdr=0.0000  coef=C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c
  cogency    p_raw=0.1853  p_fdr=0.2382  coef=C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c
  cogency    p_raw=0.0000  p_fdr=0.0000  coef=C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c
  relevance  p_raw=0.9056  p_fdr=0.9315  coef=C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c
  relevance  p_raw=0.0000  p_fdr=0.0000  coef=C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c
  dproc      p_raw=0.2721  p_fdr=0.3160  coef=C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c
  dproc      p_raw=0.0000  p_fdr=0.0000  coef=C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c

========================================================================
  COMPLEMENTARY E2 METRICS  (SCQA8_v8 Â§E2 -- separate from Stage 1/2)
========================================================================

  Per SCQA8_v8: DIS, ACA, PRR, and DeltaCC are analyzed as complementary E2
  metrics targeting responsiveness, doctrinal adherence, post-shock recovery,
  and planning-execution consistency under adversarial shocks S1-S4.
  They are NOT part of the Stage 1/2 longitudinal judge-quality core.

  Analysis scope:
    * Shock turns      : rows where shock_type_turn in {S1, S2, S3, S4}
    * Post-shock window: shock turn + up to 2 subsequent turns within the same
      debate (matching compute_kes_metrics.py _compute_prr window definition:
      scores at t, t+1, t+2 where t is the shock turn).
  This restriction is intentional: SCQA8_v8 Â§E2 binds these metrics to
  resilience under perturbation, not to general longitudinal trends.


  -- DIS (dis_v5) -- Dialectical Interactivity  [shock + post-shock window only] --

  Lightweight proxy for opponent-directed conceptual engagement.
  Reported for shock turns and the 2-turn post-shock window (t, t+1, t+2).
  SCQA8_v8 Â§E2 treats DIS as a resilience metric under shocks; longitudinal
  trends over all non-shock turns are not reported here.

  Shock-window DIS rows: 1620

  -- DIS at shock turn (window_offset=0) by condition x shock type --
anchor_shock         S1      S2      S3      S4
condition                                      
cfr_no_kg        0.0377  0.0180  0.0139  0.0381
kg_cfr_full      0.0350  0.0012  0.0487  0.0454
no_cfr_baseline  0.0067  0.0134  0.0407  0.0741

  -- DIS mean across full post-shock window (offset 0-2) by condition x shock --
anchor_shock         S1      S2      S3      S4
condition                                      
cfr_no_kg        0.0717  0.0230  0.0238  0.0560
kg_cfr_full      0.0512  0.0163  0.0235  0.0665
no_cfr_baseline  0.0295  0.0202  0.0520  0.0400

  -- DIS mean by window_offset (0=shock, 1=t+1, 2=t+2) x condition --
condition      cfr_no_kg  kg_cfr_full  no_cfr_baseline
window_offset                                         
0                 0.0265       0.0314           0.0355
1                 0.0390       0.0457           0.0183
2                 0.0614       0.0463           0.0519

  -- DIS shock-window LMM: dis_v5 ~ C(condition) + C(anchor_shock) + window_offset + C(condition):window_offset --
  (groups = debate_id; models DIS trajectory within the post-shock window)
    debates=270, scenarios=3
    [dis_sw] random-slope converged (no scenario RE)

                                                                                    Coef. Std.Err.       z  P>|z|  [0.025 0.975]
Intercept                                                                           0.031    0.013   2.466  0.014   0.006  0.056
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]                  -0.003    0.014  -0.242  0.808  -0.031  0.024
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]                 0.008    0.014   0.579  0.563  -0.020  0.036
C(anchor_shock, Treatment(reference='S1'))[T.S2]                                   -0.021    0.012  -1.802  0.072  -0.044  0.002
C(anchor_shock, Treatment(reference='S1'))[T.S3]                                   -0.012    0.012  -0.957  0.339  -0.035  0.012
C(anchor_shock, Treatment(reference='S1'))[T.S4]                                    0.007    0.012   0.571  0.568  -0.016  0.030
window_offset                                                                       0.006    0.008   0.726  0.468  -0.010  0.022
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:window_offset     0.008    0.011   0.699  0.485  -0.014  0.030
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:window_offset   0.001    0.011   0.131  0.896  -0.021  0.024
Group Var                                                                           0.007    0.008                              
Group x turn_c Cov                                                                  0.002    0.004                              
turn_c Var                                                                          0.001    0.002                              
  converged=True  |  slope_only

  -- ACA -- Axiomatic Constraint Adherence  [shock + post-shock window only] --

  Operational proxy for doctrinal constraint compliance.
  Reported for shock turns and the 2-turn post-shock window (t, t+1, t+2).
  SCQA8_v8 Â§E2 treats ACA as a resilience metric; non-shock turns excluded.

  Shock-window ACA rows: 1620

  -- ACA at shock turn (window_offset=0) by condition x shock type --
anchor_shock         S1      S2      S3      S4
condition                                      
cfr_no_kg        0.0278  0.0000  0.0625  0.2941
kg_cfr_full      0.2917  0.3542  0.2424  0.3333
no_cfr_baseline  0.0000  0.0625  0.0952  0.1765

  -- ACA mean across full post-shock window by condition x shock --
anchor_shock         S1      S2      S3      S4
condition                                      
cfr_no_kg        0.2130  0.2444  0.4097  0.5229
kg_cfr_full      0.5000  0.5278  0.4545  0.5098
no_cfr_baseline  0.1966  0.3611  0.4444  0.5359

  -- ACA mean by window_offset x condition --
condition      cfr_no_kg  kg_cfr_full  no_cfr_baseline
window_offset                                         
0                 0.1056       0.3111           0.0889
1                 0.4667       0.6056           0.5444
2                 0.5111       0.5889           0.5500

  -- ACA shock-window LMM --
    debates=270, scenarios=3
    [aca_sw] random-slope did not converge; trying intercept-only
    [aca_sw] intercept-only converged=True

                                                                                    Coef. Std.Err.       z  P>|z|  [0.025  0.975]
Intercept                                                                           0.053    0.039   1.372  0.170  -0.023   0.129
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]                  -0.009    0.045  -0.194  0.846  -0.097   0.079
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]                 0.206    0.045   4.570  0.000   0.117   0.294
C(anchor_shock, Treatment(reference='S1'))[T.S2]                                    0.074    0.034   2.203  0.028   0.008   0.140
C(anchor_shock, Treatment(reference='S1'))[T.S3]                                    0.133    0.035   3.809  0.000   0.064   0.201
C(anchor_shock, Treatment(reference='S1'))[T.S4]                                    0.212    0.033   6.408  0.000   0.147   0.277
window_offset                                                                       0.231    0.024   9.702  0.000   0.184   0.277
C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:window_offset    -0.028    0.034  -0.827  0.409  -0.094   0.038
C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:window_offset  -0.092    0.034  -2.728  0.006  -0.158  -0.026
Group Var                                                                           0.006    0.009                               
  converged=True  |  intercept_only

  -- PRR -- Perturbation Rebound Rate  [inherently shock-indexed] --

  Post-shock judge-score recovery metric.
  Defined in compute_kes_metrics.py _compute_prr as:
    drop = overall(t) - overall(t-1)    [requires drop < -0.1]
    PRR  = (overall(t+2) - overall(t)) / |drop|
    rebound_1 = overall(t+1) - overall(t)
    rebound_2 = overall(t+2) - overall(t+1)
  PRR is only computed for turns with a meaningful quality drop (>0.1);
  it is therefore already a shock-window measurement by construction.

  PRR rows: 197

  -- PRR by condition --
                 count    mean     std     min     25%     50%     75%     max
condition                                                                     
cfr_no_kg         79.0  1.4712  0.8887 -0.7000  0.9354  1.0769  1.8750  4.1000
kg_cfr_full       30.0  0.5815  0.6486 -0.9091  0.1432  0.4920  1.0000  2.2000
no_cfr_baseline   88.0  1.5893  0.9306 -0.2941  1.0000  1.4200  2.0271  5.1333

  -- PRR by condition x shock_turn --
shock_turn           S1      S2      S3      S4
condition                                      
cfr_no_kg        1.6385  1.4741  1.7327  1.1397
kg_cfr_full      0.6644  0.4409  0.3952  0.6612
no_cfr_baseline  2.0959  1.3344  1.5890  1.2109

  -- All PRR rows (shock turn, drop, and recovery trajectory) --
      condition                                  debate_id  turn_number shock_turn       prr  prr_drop  prr_rebound_1  prr_rebound_2
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_000            3         S1  1.600000     -0.50           0.85          -0.05
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_001            3         S3  2.433333     -0.30           0.65           0.08
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_002            3         S1  0.866667     -0.75           0.60           0.05
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_003            3         S4  0.933333     -0.75           0.68           0.02
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_004            3         S3  1.250000     -0.60           0.65           0.10
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_005            3         S3  0.928571     -0.70           0.65           0.00
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_007            3         S2  0.933333     -0.75           0.77          -0.07
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_008            3         S3  1.000000     -0.70           0.65           0.05
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_009            3         S4  1.145455     -0.55           0.55           0.08
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_010            3         S2  0.875000     -0.80           0.65           0.05
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_011            3         S4  1.000000     -0.75           0.65           0.10
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_012            3         S3  3.900000     -0.20           0.81          -0.03
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_013            3         S2  1.640000     -0.50           0.81           0.01
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_014            3         S1  3.750000     -0.20           0.65           0.10
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_015            3         S2  2.142857     -0.35           0.70           0.05
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_016            3         S4  0.922222     -0.90           0.80           0.03
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_018            3         S2  0.971429     -0.70           0.60           0.08
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_019            3         S1  1.460000     -0.50           0.60           0.13
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_020            3         S4  1.272727     -0.55           0.60           0.10
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_021            3         S4  1.033333     -0.60           0.45           0.17
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_022            3         S2  0.857143     -0.70           0.60           0.00
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_023            3         S2  1.093333     -0.75           0.65           0.17
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_025            3         S2  1.875000     -0.40           0.70           0.05
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_026            3         S3  0.937500     -0.80           0.70           0.05
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_027            3         S4  1.076923     -0.65           0.45           0.25
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_028            3         S2  1.366667     -0.60           0.77           0.05
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_029            3         S2  1.500000     -0.40           0.82          -0.22
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_000            3         S1  1.000000     -0.60           0.45           0.15
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_001            3         S3  0.857143     -0.70           0.55           0.05
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_002            3         S1  0.875000     -0.80           0.50           0.20
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_004            3         S3  2.933333     -0.30           0.30           0.58
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_005            3         S3  0.986667     -0.75           0.70           0.04
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_006            3         S2  2.166667     -0.30           0.70          -0.05
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_007            3         S2  1.942857     -0.35           0.67           0.01
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_009            3         S4  0.920000     -0.75           0.58           0.11
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_010            3         S2  1.440000     -0.50           0.82          -0.10
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_011            3         S4  1.971429     -0.35           0.65           0.04
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_014            3         S1  1.914286     -0.35           0.67           0.00
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_015            3         S2  1.625000     -0.40           0.68          -0.03
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_016            3         S4  1.185714     -0.70           0.82           0.01
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_017            3         S1  3.600000     -0.20           0.62           0.10
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_018            3         S2  1.375000     -0.40           0.61          -0.06
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_019            3         S1  0.960000     -0.75           0.73          -0.01
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_021            3         S4  3.500000     -0.20           0.35           0.35
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_022            3         S2  1.022222     -0.90           0.75           0.17
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_024            3         S3  3.600000     -0.20           0.71           0.01
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_025            3         S2  3.150000     -0.20           0.72          -0.09
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_026            3         S3  0.937500     -0.80           0.80          -0.05
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_027            3         S4  1.400000     -0.50           0.70           0.00
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_028            3         S2  0.800000     -0.50           0.51          -0.11
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_029            3         S2  0.977778     -0.90           0.84           0.04
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_000            3         S1  1.925000     -0.40           0.65           0.12
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_001            3         S3  4.100000     -0.20           0.55           0.27
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_002            3         S1  0.960000     -0.75           0.65           0.07
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_004            3         S3  1.600000     -0.50           0.80           0.00
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_005            3         S3  2.600000     -0.30           0.65           0.13
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_007            3         S2  0.966667     -0.60           0.45           0.13
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_009            3         S4  0.875000     -0.80           0.55           0.15
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_010            3         S2  2.333333     -0.30           0.75          -0.05
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_011            3         S4  1.400000     -0.55           0.69           0.08
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_012            3         S3  1.012500     -0.80           0.65           0.16
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_015            3         S2  0.983333     -0.60           0.43           0.16
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_016            3         S4  1.085714     -0.70           0.63           0.13
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_017            3         S1  0.750000     -0.40           0.25           0.05
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_018            3         S2  0.933333     -0.75           0.65           0.05
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_019            3         S1  1.640000     -0.50           0.75           0.07
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_020            3         S4  0.857143     -0.70           0.53           0.07
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_021            3         S4  0.812500     -0.80           0.70          -0.05
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_022            3         S2  1.000000     -0.65           0.65           0.00
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_023            3         S2  2.400000     -0.30           0.60           0.12
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_024            3         S3  0.937500     -0.80           0.63           0.12
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_025            3         S2  2.400000     -0.30           0.68           0.04
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_026            3         S3  1.875000     -0.40           0.65           0.10
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_027            3         S4  1.000000     -0.80           0.74           0.06
      cfr_no_kg       exp_e2_20260314_052549_cfr_no_kg_028            3         S2  1.028571     -0.70           0.73          -0.01
      cfr_no_kg       exp_e2_20260313_223929_cfr_no_kg_029            5         S4  0.681818     -0.22           0.13           0.02
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_010            5         S3 -0.700000     -0.10          -0.02          -0.05
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_012            5         S4  0.588235     -0.17           0.00           0.10
      cfr_no_kg       exp_e2_20260314_015807_cfr_no_kg_028            5         S4  0.272727     -0.11          -0.05           0.08
    kg_cfr_full     exp_e2_20260313_223929_kg_cfr_full_002            3         S1  0.333333     -0.12           0.05          -0.01
    kg_cfr_full     exp_e2_20260313_223929_kg_cfr_full_010            3         S4  0.000000     -0.11           0.05          -0.05
    kg_cfr_full     exp_e2_20260313_223929_kg_cfr_full_011            3         S1  0.529412     -0.17           0.20          -0.11
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_003            3         S3  0.437500     -0.16           0.05           0.02
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_011            3         S1 -0.238095     -0.21           0.10          -0.15
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_012            3         S2  0.454545     -0.11          -0.05           0.10
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_015            3         S2  0.750000     -0.16           0.05           0.07
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_018            3         S2  0.000000     -0.16           0.10          -0.10
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_025            3         S1  1.300000     -0.10           0.05           0.08
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_027            3         S3  1.347826     -0.23           0.20           0.11
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_000            3         S2  0.000000     -0.11           0.05          -0.05
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_001            3         S4  0.312500     -0.16           0.05           0.00
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_007            3         S3 -0.909091     -0.11           0.08          -0.18
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_025            3         S1 -0.454545     -0.11           0.02          -0.07
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_028            3         S4  0.849057     -0.53           0.54          -0.09
    kg_cfr_full     exp_e2_20260313_223929_kg_cfr_full_008            5         S4  0.000000     -0.11          -0.05           0.05
    kg_cfr_full     exp_e2_20260313_223929_kg_cfr_full_011            5         S1  0.272727     -0.11           0.15          -0.12
    kg_cfr_full     exp_e2_20260313_223929_kg_cfr_full_014            5         S1  2.200000     -0.10           0.05           0.17
    kg_cfr_full     exp_e2_20260313_223929_kg_cfr_full_018            5         S4  1.000000     -0.15           0.13           0.02
    kg_cfr_full     exp_e2_20260313_223929_kg_cfr_full_029            5         S1  0.333333     -0.15           0.04           0.01
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_000            5         S2  1.000000     -0.13           0.10           0.03
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_011            5         S1  1.266667     -0.15           0.10           0.09
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_014            5         S1  1.000000     -0.10           0.12          -0.02
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_021            5         S1  1.200000     -0.10           0.13          -0.01
    kg_cfr_full     exp_e2_20260314_015807_kg_cfr_full_023            5         S3  1.000000     -0.15           0.15           0.00
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_006            5         S4  1.300000     -0.10           0.13           0.00
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_007            5         S4  1.166667     -0.18           0.10           0.11
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_020            5         S1  0.294118     -0.17           0.05           0.00
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_021            5         S1  0.600000     -0.25           0.05           0.10
    kg_cfr_full     exp_e2_20260314_052549_kg_cfr_full_026            5         S3  0.100000     -0.10          -0.02           0.03
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_001            3         S1  2.600000     -0.25           0.60           0.05
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_002            3         S4  1.025000     -0.80           0.75           0.07
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_003            3         S3  2.025000     -0.40           0.70           0.11
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_005            3         S3  1.233333     -0.60           0.70           0.04
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_006            3         S3  2.400000     -0.30           0.72           0.00
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_008            3         S4  1.640000     -0.50           0.70           0.12
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_009            3         S2  0.933333     -0.75           0.60           0.10
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_010            3         S3  1.420000     -0.50           0.60           0.11
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_011            3         S2  1.026667     -0.75           0.77           0.00
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_012            3         S2  1.000000     -0.55           0.62          -0.07
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_013            3         S1  2.025000     -0.40           0.82          -0.01
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_014            3         S1  2.125000     -0.40           0.83           0.02
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_015            3         S1  2.750000     -0.20           0.61          -0.06
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_017            3         S2  1.500000     -0.40           0.65          -0.05
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_018            3         S3  1.566667     -0.60           0.80           0.14
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_019            3         S4  2.600000     -0.25           0.50           0.15
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_020            3         S2  1.000000     -0.75           0.76          -0.01
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_021            3         S4  0.937500     -0.80           0.70           0.05
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_022            3         S4  1.030769     -0.65           0.60           0.07
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_023            3         S4  1.500000     -0.50           0.64           0.11
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_024            3         S1  0.800000     -0.50           0.43          -0.03
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_025            3         S1  1.600000     -0.45           0.65           0.07
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_027            3         S1  2.150000     -0.20           0.35           0.08
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_028            3         S1  1.075000     -0.80           0.82           0.04
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_029            3         S3  1.620000     -0.50           0.77           0.04
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_000            3         S1  2.700000     -0.30           0.70           0.11
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_001            3         S1  1.533333     -0.60           0.82           0.10
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_002            3         S4  1.083333     -0.60           0.62           0.03
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_003            3         S3  3.500000     -0.20           0.70           0.00
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_004            3         S1  1.640000     -0.50           0.81           0.01
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_005            3         S3  0.906667     -0.75           0.60           0.08
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_006            3         S3  1.540000     -0.50           0.81          -0.04
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_007            3         S1  0.986667     -0.75           0.81          -0.07
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_008            3         S4  0.933333     -0.75           0.63           0.07
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_009            3         S2  1.153846     -0.65           0.70           0.05
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_010            3         S3  1.090909     -0.55           0.71          -0.11
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_011            3         S2  0.875000     -0.80           0.74          -0.04
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_012            3         S2  2.333333     -0.30           0.73          -0.03
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_013            3         S1  2.000000     -0.30           0.57           0.03
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_014            3         S1  5.000000     -0.10           0.52          -0.02
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_015            3         S1  3.650000     -0.20           0.65           0.08
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_016            3         S2  1.750000     -0.40           0.82          -0.12
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_017            3         S2  0.866667     -0.75           0.77          -0.12
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_018            3         S3  1.363636     -0.55           0.65           0.10
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_019            3         S4  1.971429     -0.35           0.55           0.14
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_020            3         S2  2.500000     -0.30           0.80          -0.05
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_021            3         S4  2.680000     -0.25           0.77          -0.10
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_022            3         S4  1.000000     -0.70           0.77          -0.07
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_023            3         S4  1.400000     -0.30           0.17           0.25
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_024            3         S1  0.933333     -0.75           0.74          -0.04
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_025            3         S1  0.933333     -0.75           0.72          -0.02
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_026            3         S4  1.533333     -0.45           0.65           0.04
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_028            3         S1  2.250000     -0.20           0.62          -0.17
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_029            3         S3  1.700000     -0.40           0.70          -0.02
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_000            3         S1  1.950000     -0.40           0.70           0.08
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_001            3         S1  3.650000     -0.20           0.73           0.00
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_002            3         S4  1.420000     -0.50           0.72          -0.01
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_003            3         S3  1.750000     -0.40           0.81          -0.11
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_005            3         S3  0.875000     -0.80           0.70           0.00
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_006            3         S3  2.175000     -0.40           0.75           0.12
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_007            3         S1  0.781818     -0.55           0.45          -0.02
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_008            3         S4  2.433333     -0.30           0.55           0.18
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_009            3         S2  0.875000     -0.80           0.70           0.00
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_010            3         S3  1.025000     -0.80           0.68           0.14
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_011            3         S2  1.280000     -0.50           0.66          -0.02
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_012            3         S2  2.733333     -0.30           0.82           0.00
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_013            3         S1  5.133333     -0.15           0.63           0.14
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_014            3         S1  2.700000     -0.30           0.70           0.11
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_015            3         S1  1.640000     -0.50           0.77           0.05
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_017            3         S2  2.033333     -0.30           0.61           0.00
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_018            3         S3  2.100000     -0.30           0.61           0.02
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_019            3         S4  1.076923     -0.65           0.30           0.40
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_020            3         S2  1.000000     -0.85           0.84           0.01
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_021            3         S4  0.925000     -0.80           0.72           0.02
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_022            3         S4  1.012500     -0.80           0.48           0.33
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_023            3         S4  1.555556     -0.45           0.50           0.20
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_025            3         S1  1.875000     -0.40           0.63           0.12
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_026            3         S4  1.142857     -0.70           0.80           0.00
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_027            3         S1  1.080000     -0.75           0.67           0.14
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_028            3         S1  1.026667     -0.75           0.70           0.07
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_029            3         S3  0.900000     -0.50           0.35           0.10
no_cfr_baseline exp_e2_20260313_223929_no_cfr_baseline_016            5         S2  0.909091     -0.11           0.05           0.05
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_010            5         S4  0.454545     -0.11           0.03           0.02
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_016            5         S2  0.250000     -0.12           0.05          -0.02
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_017            5         S4  0.000000     -0.12           0.00           0.00
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_021            5         S3  1.000000     -0.10           0.01           0.09
no_cfr_baseline exp_e2_20260314_015807_no_cfr_baseline_028            5         S4 -0.294118     -0.17           0.05          -0.10
no_cfr_baseline exp_e2_20260314_052549_no_cfr_baseline_003            5         S4  0.000000     -0.11           0.04          -0.04

  -- CC -- Counterfactual Consistency (ALIGNED score)  [shock + post-shock window] --

  *** SCOPE LIMITATION â€” DO NOT CONFLATE WITH THE SWAPPED-PLAN TEST ***

  SCQA8_v8 Â§H3.2 defines:
    DeltaCC_t = sim(v_Pt, v_Et) - E_{i!=t}[sim(v_Pt, v_Ei)]
  where v_Pt = private plan embedding, v_Et = public response embedding.

  The Swapped-Plan Test REQUIRES raw embedding vectors (v_Pt, v_Ei for i!=t).
  These are NOT stored in E2_Turns_judge_fixed.csv.

  cc_v5 = aligned similarity only (numerator of DeltaCC; no swapped baseline).
  Analysis here:
    (a) Aligned CC over the shock + post-shock window â€” characterises
        plan-execution consistency specifically during adversarial perturbation.
    (b) cc_kes vs cc_v5 cross-check (embedding version stability, all CC rows).
  Neither (a) nor (b) constitutes the Swapped-Plan Test.

  Shock-window CC rows: 1080

  -- Aligned CC at shock turn (offset=0) by condition x shock type --
anchor_shock      S1      S2      S3      S4
condition                                   
cfr_no_kg     0.1263  0.0822  0.1128  0.1077
kg_cfr_full   0.1402  0.1205  0.1215  0.1192

  -- Aligned CC mean across post-shock window by condition x shock --
anchor_shock      S1      S2      S3      S4
condition                                   
cfr_no_kg     0.1481  0.1104  0.1227  0.1130
kg_cfr_full   0.1374  0.1197  0.1296  0.1119
  Reference condition 'no_cfr_baseline' absent from CC shock-window subset (['cfr_no_kg', 'kg_cfr_full']); LMM not applicable.

  (b) cc_kes vs cc_v5 embedding robustness cross-check (all CC turns):
  N=1620
  Pearson  r=0.0375  (p=0.1311)
  Spearman r=0.1037  (p=0.0000)
  cc_kes: mean=0.7592  sd=0.0464
  cc_v5 : mean=0.1286  sd=0.0617

========================================================================
  S4 TOST -- Resilience Equivalence on Negative-Control Shock (SCQA8_v8 Â§H3.1)
========================================================================

  S4 = direct ad-hominem shock (emotional, non-logical negative control).
  SCQA8_v8 hypothesis: the KG-CFR architecture's protective effects are
  specific to logical resilience (S1-S3); on the non-logical S4 shock all
  conditions should be statistically EQUIVALENT in their resilience response.

  Primary metrics (resilience under S4):
    DIS at S4 shock turn â€” opponent-directed responsiveness under ad-hominem
    ACA at S4 shock turn â€” doctrinal constraint adherence under ad-hominem
    These directly measure process-level resilience behavior.

  Secondary metrics (judge dimensions at S4 turn, for completeness):
    overall, clarity, cogency, relevance
    These are included as supporting context, not as the primary resilience test.

  Equivalence margin: +-0.5 pooled SD.  Test: Two One-Sided Tests (Welch).
  Data: S4 shock turns only (shock_type_turn == 'S4').

  S4 judge rows : 153 | {'no_cfr_baseline': 51, 'cfr_no_kg': 51, 'kg_cfr_full': 51}
  S4 all rows   : 153 | {'no_cfr_baseline': 51, 'cfr_no_kg': 51, 'kg_cfr_full': 51}

  -- PRIMARY â€” DIS at S4 shock turn --

  DIS_S4: cfr_no_kg vs no_cfr_baseline
    n1=51  n2=51  diff=-0.0360  pooled_sd=0.1981  delta=+-0.0990
    p_lower=0.0557  p_upper=0.0004  p_TOST=0.0557  ->  not equivalent at alpha=0.05

  DIS_S4: kg_cfr_full vs no_cfr_baseline
    n1=51  n2=51  diff=-0.0287  pooled_sd=0.1834  delta=+-0.0917
    p_lower=0.0433  p_upper=0.0007  p_TOST=0.0433  ->  EQUIVALENT at alpha=0.05

  -- PRIMARY â€” ACA at S4 shock turn --

  ACA_S4: cfr_no_kg vs no_cfr_baseline
    n1=51  n2=51  diff=+0.1176  pooled_sd=0.4243  delta=+-0.2121
    p_lower=0.0001  p_upper=0.1318  p_TOST=0.1318  ->  not equivalent at alpha=0.05

  ACA_S4: kg_cfr_full vs no_cfr_baseline
    n1=51  n2=51  diff=+0.1569  pooled_sd=0.4330  delta=+-0.2165
    p_lower=0.0000  p_upper=0.2443  p_TOST=0.2443  ->  not equivalent at alpha=0.05

  -- SECONDARY â€” judge dimensions at S4 (overall, clarity, cogency, relevance) --
  [Supporting context only, not the primary resilience test]

  overall_S4: cfr_no_kg vs no_cfr_baseline
    n1=51  n2=51  diff=+0.0004  pooled_sd=0.3501  delta=+-0.1751
    p_lower=0.0065  p_upper=0.0067  p_TOST=0.0067  ->  EQUIVALENT at alpha=0.05

  overall_S4: kg_cfr_full vs no_cfr_baseline
    n1=51  n2=51  diff=+0.2390  pooled_sd=0.2546  delta=+-0.1273
    p_lower=0.0000  p_upper=0.9846  p_TOST=0.9846  ->  not equivalent at alpha=0.05

  clarity_S4: cfr_no_kg vs no_cfr_baseline
    n1=51  n2=51  diff=+0.0029  pooled_sd=0.3318  delta=+-0.1659
    p_lower=0.0058  p_upper=0.0074  p_TOST=0.0074  ->  EQUIVALENT at alpha=0.05

  clarity_S4: kg_cfr_full vs no_cfr_baseline
    n1=51  n2=51  diff=+0.2078  pooled_sd=0.2398  delta=+-0.1199
    p_lower=0.0000  p_upper=0.9653  p_TOST=0.9653  ->  not equivalent at alpha=0.05

  cogency_S4: cfr_no_kg vs no_cfr_baseline
    n1=51  n2=51  diff=-0.0022  pooled_sd=0.3333  delta=+-0.1667
    p_lower=0.0072  p_upper=0.0060  p_TOST=0.0072  ->  EQUIVALENT at alpha=0.05

  cogency_S4: kg_cfr_full vs no_cfr_baseline
    n1=51  n2=51  diff=+0.2202  pooled_sd=0.2425  delta=+-0.1213
    p_lower=0.0000  p_upper=0.9781  p_TOST=0.9781  ->  not equivalent at alpha=0.05

  relevance_S4: cfr_no_kg vs no_cfr_baseline
    n1=51  n2=51  diff=-0.0033  pooled_sd=0.3718  delta=+-0.1859
    p_lower=0.0074  p_upper=0.0058  p_TOST=0.0074  ->  EQUIVALENT at alpha=0.05

  relevance_S4: kg_cfr_full vs no_cfr_baseline
    n1=51  n2=51  diff=+0.2661  pooled_sd=0.2684  delta=+-0.1342
    p_lower=0.0000  p_upper=0.9918  p_TOST=0.9918  ->  not equivalent at alpha=0.05

========================================================================
  EXPORT
========================================================================
  Coefficient table -> C:\03_Studies\04_Thesis\philosopher-agents-conversations\analysis\outputs\e2\e2_scqa8_coefs.csv  (40 rows)
  FDR table         -> C:\03_Studies\04_Thesis\philosopher-agents-conversations\analysis\outputs\e2\e2_scqa8_fdr.csv  (36 rows)
  TOST table        -> C:\03_Studies\04_Thesis\philosopher-agents-conversations\analysis\outputs\e2\e2_scqa8_tost.csv  (12 rows)
```

### fdr.csv (full)
```csv
dv,coef,p_raw,p_fdr,sig_fdr
cogency,turn_c,1.7808287127998036e-250,6.410983366079293e-249,True
relevance,turn_c,6.831146572329941e-198,1.2296063830193895e-196,True
clarity,turn_c,6.753460405935813e-151,8.104152487122976e-150,True
cogency,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c",2.1954500338242178e-141,1.975905030441796e-140,True
relevance,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c",2.4089523195531143e-108,1.7344456700782424e-107,True
dproc,turn_c,3.6018785967482536e-100,2.1611271580489525e-99,True
clarity,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c",1.255086166733221e-86,6.454728857485137e-86,True
relevance,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]",1.4287429247092952e-55,6.429343161191829e-55,True
cogency,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]",2.529411890297566e-55,1.0117647561190264e-54,True
dproc,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]",2.8310848530195826e-49,1.0191905470870497e-48,True
dproc,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c",3.3845190333721754e-47,1.1076607745581663e-46,True
clarity,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]",6.936513924674762e-34,2.080954177402429e-33,True
cogency,"C(shock_turn, Treatment(reference='none'))[T.S1]",2.7134472308853932e-11,7.514161562451858e-11,True
relevance,"C(shock_turn, Treatment(reference='none'))[T.S1]",1.0571386056298106e-09,2.7183564144766556e-09,True
clarity,"C(shock_turn, Treatment(reference='none'))[T.S1]",7.953552234885992e-07,1.908852536372638e-06,True
cogency,"C(shock_turn, Treatment(reference='none'))[T.S2]",3.262892290075683e-05,7.341507652670286e-05,True
relevance,"C(shock_turn, Treatment(reference='none'))[T.S2]",3.7394451506684845e-05,7.918825024945026e-05,True
clarity,"C(shock_turn, Treatment(reference='none'))[T.S2]",0.0004096337425480418,0.0008192674850960836,True
relevance,"C(shock_turn, Treatment(reference='none'))[T.S3]",0.004021566686580166,0.00761981056404663,True
cogency,"C(shock_turn, Treatment(reference='none'))[T.S4]",0.007295814792917991,0.013132466627252383,True
clarity,"C(shock_turn, Treatment(reference='none'))[T.S4]",0.010700161216548468,0.018343133514083086,True
dproc,"C(shock_turn, Treatment(reference='none'))[T.S4]",0.015152013240578457,0.024794203484582927,True
clarity,"C(shock_turn, Treatment(reference='none'))[T.S3]",0.020848211014340735,0.03263198245722898,True
relevance,"C(shock_turn, Treatment(reference='none'))[T.S4]",0.023593139311719098,0.03538970896757865,True
cogency,"C(shock_turn, Treatment(reference='none'))[T.S3]",0.03306602653583186,0.04761507821159788,True
dproc,"C(shock_turn, Treatment(reference='none'))[T.S3]",0.05282221167530408,0.07313844693503642,False
clarity,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]",0.10960780742255782,0.1461437432300771,False
cogency,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c",0.18527962554547384,0.2382166614156092,False
dproc,"C(shock_turn, Treatment(reference='none'))[T.S1]",0.23462131358630567,0.2912540444519656,False
dproc,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c",0.27211775085506007,0.3160077106703923,False
dproc,"C(shock_turn, Treatment(reference='none'))[T.S2]",0.26636113409462525,0.3160077106703923,False
relevance,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]",0.43599567242340975,0.4853172446375248,False
clarity,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c",0.44487414091773103,0.4853172446375248,False
dproc,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]",0.5029703975289916,0.5325568915012853,False
relevance,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c",0.9056063338202915,0.9314808005008713,False
cogency,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]",0.9614377025076236,0.9614377025076236,False
```

### Condition * Turn interaction coefficients (from coefs.csv)
```csv
dv,coef,estimate,se,z,p_raw,p_fdr,re_type
clarity,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c",0.0020004862215235,0.0026184771366079,0.76398842424686,0.444874140917731,0.4853172446375248,intercept_only
clarity,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c",-0.0516403706688475,0.002617700402385,-19.727380040052044,1.255086166733221e-86,6.454728857485137e-86,intercept_only
cogency,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c",-0.003284454231841,0.0024794452540114,-1.324673019711645,0.1852796255454738,0.2382166614156092,intercept_only
cogency,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c",-0.062748063844508,0.0024787112993064,-25.314793159681784,2.1954500338242178e-141,1.975905030441796e-140,intercept_only
relevance,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c",0.0003303162557515,0.0027855456801709,0.1185822433654439,0.9056063338202917,0.9314808005008712,slope_only
relevance,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c",-0.0615765622486813,0.0027847248501824,-22.112260837779505,2.4089523195531147e-108,1.7344456700782424e-107,slope_only
dproc,"C(condition, Treatment(reference='no_cfr_baseline'))[T.cfr_no_kg]:turn_c",0.0276138895545718,0.0251447124622738,1.098198660891534,0.27211775085506,0.3160077106703923,intercept_only
dproc,"C(condition, Treatment(reference='no_cfr_baseline'))[T.kg_cfr_full]:turn_c",-0.3641472082655359,0.0252366341721482,-14.429309621146626,3.3845190333721754e-47,1.1076607745581663e-46,intercept_only
```
