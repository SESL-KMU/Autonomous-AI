function sys = ref_yawrate(u)

% u(1) : body velocity at x axis [m/sec]
% u(2) : steering wheel angle [rad/sec]
% lf : front wheel base [m]
% lr : rear wheel base [m]
% CF0, CR0 : conering coefficient  [m/sec]

LF = 0.998;
LR = 1.572;
M_BODY  =  1395;

CF0  =  36500.0;	    
CR0  =  63000.0	;


Vch= 2*CF0*CR0*(LF+LR)^2/((LR*CR0-LF*CF0)*M_BODY);

speed= u(2);
steer =u(1);

%sys = speed / ((lf + lr) * (1 + speed.^2 / vc.^2)) * steer;



sys = speed/((LF+LR)*(1+speed^2/Vch))*steer;