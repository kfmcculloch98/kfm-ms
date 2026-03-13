ptf ~
param_sets:
- name: Low T, Low S
  t_eval: 365.0
  T: 1e-7
  S: 1e-5
- name: High T, Low S
  t_eval: 365.0
  T: 1e-3
  S: 1e-5
- name: Low T, High S
  t_eval: 365.0
  T: 1e-7
  S: 1e-1
- name: High T, High S
  t_eval: 365.0
  T: 1e-3
  S: 1e-1
grid:
  nrow: 50
  ncol: 80
  x0: 0.0
  y0: 0.0
  delr: 100.0
  delc: 100.0
T: 1e-7
S: 1e-5
t_start: 1.0
t_end: 365.0
t_eval: 365.0
obs_points_file: obs_points.csv
well_1:
  Q: ~     well_1__q      ~
  r_coord: 10
  c_coord: 20
  radial_dist: 2236.07
well_2:
  Q: ~     well_2__q      ~
  r_coord: 20
  c_coord: 60
  radial_dist: 6324.56
