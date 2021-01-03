function [ y ] = acc( u )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input
% u(1:4) : slip_ratio
% u(5) : longitudinal velocity  (m/s)
% u(6) : distance of between vehicle  (m)
% u(7) : Previous distance of front vehicle  (m)
% u(8) : Assist steering angle for integrated algorithm  (rad)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output
% y(1:4) : Pressure : acc에서 내보내는 네바퀴의 브레이크 압력
% y(5) : distance of front vehicle  (m)
% y(6) : headway time distance (m)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prameter
% delta_t_vdc : difference of speed between vehicles (m/s)
% t_h : headway time (sec)
% d_h : headway time distance (m)
% gain : weight factor
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

delta_t_vdc = 500;
slip_ratio(1:4) = u(1:4);
v = u(5);
d = u(6);
d_temp = u(7);
lks_steer = u(8);


t_h = 2;
d_h = v * t_h;
d_m = 0.65 * d_h;

gain = 25;
   
if(lks_steer == 0)  % LKS OFF
    if (d < d_h)
        if(d > d_temp)
            pressure(1) = 0;
            pressure(2) = 0;
            pressure(3) = 0;
            pressure(4) = 0;
        else
            pressure(1) = gain * (d_h - d);
            pressure(2) = gain * (d_h - d);
            pressure(3) = gain * (d_h - d);
            pressure(4) = gain * (d_h - d);
        end
    else
         if(v<=27) %설정속도 100km/h (=27.7m/s)
            pressure(1) = -(gain*3);
            pressure(2) = -(gain*3);
            pressure(3) = -(gain*3);
            pressure(4) = -(gain*3);
         else
            pressure(1) = 0;
            pressure(2) = 0;
            pressure(3) = 0;
            pressure(4) = 0;
         end
    end
    test = 0;
    
else    % LKS ON_ 동작
    if (d < d_m)
        if(d > d_temp)
            pressure(1) = 0;
            pressure(2) = 0;
            pressure(3) = 0;
            pressure(4) = 0;
        else
            pressure(1) = (gain*3) * (d_m - d);
            pressure(2) = (gain*3) * (d_m - d);
            pressure(3) = (gain*3) * (d_m - d);
            pressure(4) = (gain*3) * (d_m - d);
        end
    elseif(d_m < d < d_h)
            pressure(1) = 0;
            pressure(2) = 0;
            pressure(3) = 0;
            pressure(4) = 0;        
    else
         if(v<=27) %설정속도 100km/h (=27.7m/s)
            pressure(1) = -(gain*3);
            pressure(2) = -(gain*3);
            pressure(3) = -(gain*3);
            pressure(4) = -(gain*3);
         else
            pressure(1) = 0;
            pressure(2) = 0;
            pressure(3) = 0;
            pressure(4) = 0;
         end
    end
    test = 1;
end
    
    d_temp= d;
 
    for i = 1:4
        if slip_ratio(i) > 0.10
            pressure(i) = pressure(i) - delta_t_vdc;
            if pressure(i) < 0
                pressure(i) = 0;
            end
        end
    end

    
y(1:4) = pressure(1:4);
y(5) = d;
y(6) = d_h;
y(7) = d_m;
y(8) = test;
    