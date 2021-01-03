function [ y ] = lks( u )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input
% u(1) : x global position of vehicle  (m)
% u(2) : y global position of vehicle  (m)
% %u(3) : steering angle of vehicle (rad)
% u(3:4) : Road model x, y
% u(5) : privious x global position of vehicle  (m)
% u(6) : privious y global position of vehicle  (m)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output
% y(1) : assist steering angle of LKS (rad)
% y(2) : e1 (m)
% y(3) : e2 (m)
% y(4) : look ahead x global position of vehicle  (m)
% y(5) : look ahead y global position of vehicle  (m)
% y(6) : feedback x global position of vehicle  (m)
% y(7) : feedback y global position of vehicle  (m)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prameter
% ls : look ahead distance (m)
% ldp : lane detection permission (m)
% gain : weight factor (m)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_global = u(1);
y_global = u(2);
road(1:2) = u(3:4);
x_preglobal = u(5);
y_preglobal = u(6);

ls = 2;
ldp = 0.6;

gain1 = 0.009;
gain2 = 0.011;
gain3 = 0.020;
gain4 = 0.025;
gain5 = 0.030;
gain6 = 0.035;
gain7 = 0.040;
gain8 = 0.045;
gain9 = 0.050;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% e1 calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (x_global <= 200)
        e1 = -(y_global);
        y_d2 = 0;
        
    else
        % 현재차량 좌표에서 y축에 수직인 도로 좌표 
        x_d1 = 60* pi*  acos((y_global+100)/100)+ 200;
        y_d1 = y_global;
        
        % 현재차량 좌표에서 x축에 수직인 도로 좌표 
        x_d2 = x_global;
        y_d2 = 100* cos((x_global-200)/(60*pi))- 100;
        
        % 위 두 점을 지나는 직선에서 점(x_global, y_global)까지의 최소거리 e1 계산
        e1_den = sqrt( ((y_d2-y_d1)/(x_d2-x_d1))^2 +1 );
        e1_num = abs( ((y_d2-y_d1)/(x_d2-x_d1))*x_global - y_global + (y_d1-((y_d2-y_d1)/(x_d2-x_d1))*x_d1) );
        e1 = e1_num/e1_den;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% e2 calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(x_global == x_preglobal)
        if(y_global == y_preglobal) % (x_global,y_global)=(x_preglobal,y_preglobal) ex)초기조건(종방향 출발)
            x_vl = x_global + ls;
            y_vl = y_global;
        else                        % x축에 수직으로 횡방향 운전
            x_vl = x_global;
            y_vl = y_gloval - ls;
        end
    else
        % 현재차량 주행방향의 기울기
        a = (y_global-y_preglobal)/(x_global-x_preglobal);

        % A*(x_vl)^2 - 2*B*(x_vl) + C = 0
        A = 1+a^2;
        B = (1+a^2)*x_global;
        C = (1+a^2)*x_global^2 - ls^2;
        
        % 현재차량 좌표에서 주행방향으로 길이 ls를 갖는 주행예상 좌표
        if(A==0)    % - 2*B*(x_vl) + C = 0
            x_vl = C/(2*B);
        else        % A*(x_vl)^2 - 2*B*(x_vl) + C = 0
            x_vl = ( B+ sqrt(B^2 - A*C) )/ A;   % 근의 공식 적용
        end
        y_vl = a*(x_vl - x_global) + y_global;
    end
        
        if(x_vl <= 200)
            e2 = -(y_vl);
        else
            % 주행예상 좌표에서 y축에 수직인 도로 좌표
            x_d1vl = 60* pi*  acos((y_vl+100)/100)+ 200;
            y_d1vl = y_vl;
        
            % 주행예상 좌표에서 x축에 수직인 도로 좌표 
            x_d2vl = x_vl;
            y_d2vl = 100* cos((x_vl-200)/(60*pi))- 100;
        
            % 위 두 점을 지나는 직선에서 점(x_vl, y_vl)까지의 최소거리 e2 계산
            e2_den = sqrt( ((y_d2vl-y_d1vl)/(x_d2vl-x_d1vl))^2 +1 );
            e2_num = abs( ((y_d2vl-y_d1vl)/(x_d2vl-x_d1vl))*x_vl - y_vl + (y_d1vl-((y_d2vl-y_d1vl)/(x_d2vl-x_d1vl))*x_d1vl) );
            e2 = e2_num/e2_den;
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LKS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(e1 >= ldp)    % LKS ON
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        if((x_global - x_preglobal) == 0)   % x축에 수직으로 횡방향 주행시
            
            if( (y_global - y_preglobal) > 0 )  % y (+)방향 주행

                if( abs(e1)-abs(e2) < 0 )   % 도로 바깥으로 기울어진 경우
                
                    if( x_global > x_d1 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain2;
                    elseif( x_global < x_d1 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain2;
                    else  % x_global == x_d1  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                
                elseif( abs(e1)-abs(e2) == 0 )   % 도로와 평행한 경우
                
                    if( x_global > x_d1 )   % 도로와 평행한 경우- 도로의 우측
                        steer_assist = + gain1;
                    elseif( x_global < x_d1 )   % 도로와 평행한 경우- 도로의 좌측
                        steer_assist = - gain1;
                    else  % x_global == x_d1  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                
                else  % abs(e1)-abs(e2) > 0  % 도로 안으로 기울어진 경우
                    steer_assist = 0;
                end
            
            elseif( (y_global - y_preglobal) < 0 )  % y (-)방향 주행
                
                if( abs(e1)-abs(e2) < 0 )   % 도로 바깥으로 기울어진 경우
                
                    if( x_global < x_d1 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain2;
                    elseif( x_global > x_d1 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain2;
                    else  % x_global == x_d1  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                
                elseif( abs(e1)-abs(e2) == 0 )   % 도로와 평행한 경우
                
                    if( x_global < x_d1 )   % 도로와 평행한 경우- 도로의 우측
                        steer_assist = + gain1;
                    elseif( x_global > x_d1 )   % 도로와 평행한 경우- 도로의 좌측
                        steer_assist = - gain1;
                    else  % x_global == x_d1  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                
                else  % abs(e1)-abs(e2) > 0  % 도로 안으로 기울어진 경우
                    steer_assist = 0;
                end
                
            else    % 정지
                steer_assist = 0;
            end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                 
        else    % ((x_global - x_preglobal) != 0)
            
            if( abs(e1)-abs(e2) <= -0.002 )   % 도로 바깥으로 기울어진 경우
                
                if( 0.6 <= e1 < 0.7 )
                    if( y_global < y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain2;
                    elseif( y_global > y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain1;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                elseif( 0.7 <= e1 < 0.75 )
                    if( y_global < y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain3;
                    elseif( y_global > y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain2;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                elseif( 0.75 <= e1 < 0.8 )
                    if( y_global < y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain4;
                    elseif( y_global > y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain3;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                elseif( 0.8 <= e1 < 0.85 )
                    if( y_global < y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain5;
                    elseif( y_global > y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain3;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                elseif( 0.85 <= e1 < 0.9 )
                    if( y_global < y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain6;
                    elseif( y_global > y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain4;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                elseif( 0.9 <= e1 < 0.95 )
                    if( y_global < y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain7;
                    elseif( y_global > y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain5;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                elseif( 0.95 <= e1 < 1 )
                    if( y_global < y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain8;
                    elseif( y_global > y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain6;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                else    % (e1 >= 1)
                    if( y_global < y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 우측
                        steer_assist = + gain9;
                    elseif( y_global > y_d2 )   % 도로 바깥으로 기울어진 경우- 도로의 좌측
                        steer_assist = - gain7;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                end
              
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            elseif( -0.002< abs(e1)-abs(e2) <= 0 )   % 도로와 평행한 경우

                if( 0.6 <= e1 < 0.7 )
                    if( y_global < y_d2 )   % 도로와 평행한 경우- 도로의 우측
                        steer_assist = + gain1;
                    elseif( y_global > y_d2 )   % 도로와 평행한 경우- 도로의 좌측
                        steer_assist = - gain1;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                elseif( 0.7 <= e1 < 0.8 )
                    if( y_global < y_d2 )   % 도로와 평행한 경우- 도로의 우측
                            steer_assist = + gain2;
                    elseif( y_global > y_d2 )   % 도로와 평행한 경우- 도로의 좌측
                            steer_assist = - gain2;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                            steer_assist = 0;
                    end  
                elseif( 0.8 <= e1 < 0.9 )
                    if( y_global < y_d2 )   % 도로와 평행한 경우- 도로의 우측
                        steer_assist = + gain3;
                    elseif( y_global > y_d2 )   % 도로와 평행한 경우- 도로의 좌측
                        steer_assist = - gain3;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                elseif( 0.9<= e1 < 1 )
                    if( y_global < y_d2 )   % 도로와 평행한 경우- 도로의 우측
                        steer_assist = + gain4;
                    elseif( y_global > y_d2 )   % 도로와 평행한 경우- 도로의 좌측
                        steer_assist = - gain4;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                    
                else    % (e1 >= 1)
                    if( y_global < y_d2 )   % 도로와 평행한 경우- 도로의 우측
                        steer_assist = + gain5;
                    elseif( y_global > y_d2 )   % 도로와 평행한 경우- 도로의 좌측
                        steer_assist = - gain5;
                    else  % y_global == y_d2  % 도로 중앙에 위치한 경우
                        steer_assist = 0;
                    end
                end
                
            else  % abs(e1)-abs(e2) > 0  % 도로 안으로 기울어진 경우
                steer_assist = 0;
            end
       end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
        
    else    % LKS OFF
        steer_assist = 0;
    end
    
    x_preglobal = x_global;
    y_preglobal = y_global;

    
y(1) = steer_assist;      
y(2) = e1;  
y(3) = e2;
y(4) = x_vl;    
y(5) = y_vl;
y(6) = x_global;
y(7) = y_global; 
   