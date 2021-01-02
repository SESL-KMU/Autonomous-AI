uw=0.0; u0=30.0; rho=1.202;

%Controlled vehicle parameters:
mc=1000.0; Cdc=0.5; Arc=1.5; fc=0.015;
Kc=(1/(rho*Cdc*Arc*(u0+uw))); Tc=mc*Kc;

% 4-state system for controller-design:
Aa=[0 -1 0 0; 0 -1/Tc 0 0; 1 0 0 0; 0 0 1 0];
Ba=[0;Kc/Tc;0;0];
Ca=[1 0 0 0];

pc = [roots([1 2*0.9*0.4 0.4^2]); -1.08; -1.18];
K = place(Aa,Ba,pc);

Ac=[0 -1 0 0;
    -K(1)*Kc/Tc -(1+K(2)*Kc)/Tc -K(3)*Kc/Tc -K(4)*Kc/Tc;
    1 0 0 0;
    0 0 1 0];
Bc=[0;Kc/Tc;0;0];
Cc=[1 0 0 0];

[numc, denc]=ss2tf(Ac, Bc, Cc, 0);
sysc=tf(numc, denc);
rlocus(sysc);
eig(sysc)