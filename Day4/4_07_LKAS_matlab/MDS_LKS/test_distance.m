function [ y ] = test_distance( u )

d_init = 300;
delta_t = 0.005;
del_v = u(1);
del_vv = del_v + (v-v_f);

d = d_init - (del_vv)*delta_t;

y(1) = d;