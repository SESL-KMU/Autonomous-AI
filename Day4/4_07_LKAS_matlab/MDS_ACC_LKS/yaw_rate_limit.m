function sys = yaw_rate_limit(u)


MU = 0.3;
G = 9.81;


speed= u(1);
yaw_nominal = abs(u(2));

limit = MU * G / speed ;

if yaw_nominal <= limit
         
  sys = u(2);
  
end
 
           
if yaw_nominal > limit
           
           if u(2) < 0
          
               sys = -limit;
               
           else  
               
               sys = limit ;
               
           end
             
       end


