"""2024-08-22: login node rebooted due to security updates

===============================================================================
Last login: Wed Sep 25 23:39:53 2024 from 89.64.95.85
[athena][plgmstefaniak@login01 ~]$ sacct
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
944651       medium_up+ plgrid-gp+ plgllmeff+         16    TIMEOUT      0:0 
944651.batch      batch            plgllmeff+         16  CANCELLED     0:15 
944652       medium_up+ plgrid-gp+ plgllmeff+         16    TIMEOUT      0:0 
944652.batch      batch            plgllmeff+         16  CANCELLED     0:15 
944655       medium_up+ plgrid-gp+ plgllmeff+         16    TIMEOUT      0:0 
944655.batch      batch            plgllmeff+         16  CANCELLED     0:15 
944656       medium_up+ plgrid-gp+ plgllmeff+         16    TIMEOUT      0:0 
944656.batch      batch            plgllmeff+         16  CANCELLED     0:15 
944756       base_lr_g+ plgrid-gp+ plgllmeff+         16  COMPLETED      0:0 
944756.batch      batch            plgllmeff+         16  COMPLETED      0:0 
944757       base_lr_g+ plgrid-gp+ plgllmeff+         16  COMPLETED      0:0 
944757.batch      batch            plgllmeff+         16  COMPLETED      0:0 
944758       base_lr_g+ plgrid-gp+ plgllmeff+         16  COMPLETED      0:0 
944758.batch      batch            plgllmeff+         16  COMPLETED      0:0 
944759       base_lr_g+ plgrid-gp+ plgllmeff+         16  COMPLETED      0:0 
944759.batch      batch            plgllmeff+         16  COMPLETED      0:0 
944760       base_lr_g+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944760.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944761       base_lr_g+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944761.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944762       base_lr_g+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944762.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944763       base_lr_g+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944763.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944817       medium_re+ plgrid-gp+ plgllmeff+         32    TIMEOUT      0:0 
944817.batch      batch            plgllmeff+         32  CANCELLED     0:15 
944821       medium_re+ plgrid-gp+ plgllmeff+         32    TIMEOUT      0:0 
944821.batch      batch            plgllmeff+         32  CANCELLED     0:15 
944822       medium_re+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944822.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944824       medium_re+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944824.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944825       medium_re+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944825.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944826       medium_re+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944826.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944827       medium_re+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944827.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944828       medium_re+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944828.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944829       medium_re+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944829.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944846       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944846.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944847       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944847.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944848       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944848.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944849       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944849.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944850       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944850.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944851       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944851.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944852       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944852.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944853       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944853.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944854       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944854.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944855       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944855.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944856       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944856.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944857       medium_co+ plgrid-gp+ plgllmeff+         32  COMPLETED      0:0 
944857.batch      batch            plgllmeff+         32  COMPLETED      0:0 
944874       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944874.batch      batch            plgllmeff+         32    RUNNING      0:0 
944875       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944875.batch      batch            plgllmeff+         32    RUNNING      0:0 
944876       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944876.batch      batch            plgllmeff+         32    RUNNING      0:0 
944877       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944877.batch      batch            plgllmeff+         32    RUNNING      0:0 
944878       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944878.batch      batch            plgllmeff+         32    RUNNING      0:0 
944879       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944879.batch      batch            plgllmeff+         32    RUNNING      0:0 
944884       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944884.batch      batch            plgllmeff+         32    RUNNING      0:0 
944885       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944885.batch      batch            plgllmeff+         32    RUNNING      0:0 
944888       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944888.batch      batch            plgllmeff+         32    RUNNING      0:0 
944894       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944894.batch      batch            plgllmeff+         32    RUNNING      0:0 
944896       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944896.batch      batch            plgllmeff+         32    RUNNING      0:0 
944897       base_remo+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944897.batch      batch            plgllmeff+         32    RUNNING      0:0 
944920       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944920.batch      batch            plgllmeff+         32    RUNNING      0:0 
944921       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944921.batch      batch            plgllmeff+         32    RUNNING      0:0 
944922       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944922.batch      batch            plgllmeff+         32    RUNNING      0:0 
944925       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944925.batch      batch            plgllmeff+         32    RUNNING      0:0 
944926       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944926.batch      batch            plgllmeff+         32    RUNNING      0:0 
944927       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944927.batch      batch            plgllmeff+         32    RUNNING      0:0 
944931       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944931.batch      batch            plgllmeff+         32    RUNNING      0:0 
944932       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944932.batch      batch            plgllmeff+         32    RUNNING      0:0 
944933       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944933.batch      batch            plgllmeff+         32    RUNNING      0:0 
944934       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944934.batch      batch            plgllmeff+         32    RUNNING      0:0 
944935       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944935.batch      batch            plgllmeff+         32    RUNNING      0:0 
944936       base_cont+ plgrid-gp+ plgllmeff+         32    RUNNING      0:0 
944936.batch      batch            plgllmeff+         32    RUNNING      0:0 """
