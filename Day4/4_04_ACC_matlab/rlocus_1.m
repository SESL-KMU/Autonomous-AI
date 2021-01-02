uw=0.0; u0=30.0; rho=1.202;

%Controlled vehicle parameters:
mc=1000.0; Cdc=0.5; Arc=1.5; fc=0.015;
Kc=(1/(rho*Cdc*Arc*(u0+uw))); Tc=mc*Kc;

% 4-state system for controller-design:
Aa=[0 -1 0 0; 0 -1/Tc 0 0; 1 0 0 0; 0 0 1 0];
Ba=[0;Kc/Tc;0;0];
Ca=[1 0 0 0];
[num, den]=ss2tf(Aa, Ba, Ca, 0);
sys=tf(num, den);
rlocus(sys);
eig(sys)