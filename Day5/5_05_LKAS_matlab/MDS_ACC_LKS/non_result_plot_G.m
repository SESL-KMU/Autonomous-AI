% ******************** Output variables ***************************
%   y(0) : vehicle longitudinal velocity [m/s]           
%   y(1) : vehicle lateral velocity [m/s]
%   y(2) : x position (global coordinate) [m]
%   y(3) : y position (global coordinate) [m]
%   y(4) : x position (local coordinate) [m]
%   y(5) : y position (local coordinate) [m]
%   y(6) : front wheel steer angle [rad]
%   y(7) : yaw velocity [rad/sec]
%   y(8) : yaw angle [rad]
%   y(9) : FL slip angle [rad]
%   y(10) : FR slip angle [rad]
%   y(11) : RL slip angle [rad]
%   y(12) : RR slip angle [rad]
%   y(13) : FL wheel speed [m/s]
%   y(14) : FR wheel speed [m/s]
%   y(15) : RL wheel speed [m/s]
%   y(16) : RR wheel speed [m/s]                
%   y(17) : FL longitudinal tire force [N]
%   y(18) : FR longitudinal tire force [N]
%   y(19) : RL longitudinal tire force [N]
%   y(20) : RR longitudinal tire force [N]
%   y(21) : FL lateral tire force [N]
%   y(22) : FR lateral tire force [N]
%   y(23) : RL lateral tire force [N]
%   y(24) : RR lateral tire force [N]
%   y(25) : sideslip angle of CG [rad]
%   y(26) : Longitudinal acceleration [m/s^2]
%   y(27) : Laterall accleration [m/s^2]
%   y(28) : FL brake torque
%   y(29) : FR brake torque
%   y(30) : RL brake torque
%   y(31) : RR brake torque
% *****************************************************************


close all

load('steer.mat');
load('brake.mat');

tout = steer(1,:);
results1 = steer;
results2 = brake;
s1 = 'r--';
s2 = 'k-';
s3 = 'b-.';
s4 = 'b:';
s5 = 'g-';
sline = 'LineWidth';


figure(1)

plot(tout, results1(2,:)*3.6, s1, sline, 2); hold on;
plot(tout, results2(2,:)*3.6, s2, sline, 2); 
title('vehicle longitudinal velocity [km/s] ');
xlabel('Time[sec]');
ylabel('longitudinal velocity [km/s] ');
legend('steer ','brake');

figure(2)
subplot(2, 2, 1)
plot(tout, results1(15,:)*3.6, s1, sline, 2); hold on;
plot(tout, results2(15,:)*3.6, s2, sline, 2); 
title('FL wheel longitudinal velocity [km/s] ');
xlabel('Time[sec]');
ylabel('longitudinal velocity [km/s] ');
legend('steer ','brake');

subplot(2, 2, 2)
plot(tout, results1(16,:)*3.6, s1, sline, 2); hold on;
plot(tout, results2(16,:)*3.6, s2, sline, 2); 
title('FR wheel longitudinal velocity [km/s] ');
xlabel('Time[sec]');
ylabel('longitudinal velocity [km/s] ');
legend('steer ','brake');

subplot(2 ,2 ,3)
plot(tout, results1(17,:)*3.6, s1, sline, 2); hold on;
plot(tout, results2(17,:)*3.6, s2, sline, 2); 
title('RL wheel longitudinal velocity [km/s] ');
xlabel('Time[sec]');
ylabel('longitudinal velocity [km/s] ');
legend('steer ','brake');

subplot(2, 2, 4)
plot(tout, results1(18,:)*3.6, s1, sline, 2); hold on;
plot(tout, results2(18,:)*3.6, s2, sline, 2); 
title('RR wheel longitudinal velocity [km/s] ');
xlabel('Time[sec]');
ylabel('longitudinal velocity [km/s] ');
legend('steer ','brake');


figure(3)
plot(tout, results1(9,:)*180/pi, s1, sline, 2); hold on;
plot(tout, results2(9,:)*180/pi, s2, sline, 2); hold on;
title('yaw velocity [deg/sec]');
xlabel('Time[sec]');
ylabel('yaw velocity [deg/sec]');
legend('steer ','brake ');


figure(4)

plot(tout, results1(10,:)*180/pi, s1, sline, 2); hold on;
plot(tout, results2(10,:)*180/pi, s2, sline, 2); hold on;
title('Yaw angle of vehicle of CG');
ylabel('Yaw angle[deg]');
legend('steer ','brake ');

figure(5)
plot(tout, results1(3,:)*3.6, s1, sline, 2); hold on;
plot(tout, results2(3,:)*3.6, s2, sline, 2); hold on;
title('vehicle lateral velocity [m/s]');
ylabel('vehicle lateral velocity [m/s]');
xlabel('Time[sec]');
legend('steer ','brake ');


% Plot results(4) : x position of CG (global coordinate) [m]
%      results(5) : y position of CG (global coordinate) [m]
figure(6)
plot(results1(4,:), results1(5,:), s1, sline, 2); hold on;
plot(results2(4,:), results2(5,:), s2, sline, 2); hold on;
%plot(results3(8,:), results3(9,:), s3, sline, 2);
title('Vehicle trajectory at global coordinate');
xlabel('X-axis[m]');
ylabel('Y-axis[m]');
legend('steer ','brake ');

% Plot results(11) : FL slip angle [rad]
%   results(12) : FR slip angle [rad]
%   results(13) : RL slip angle [rad]
%   results(14) : RR slip angle [rad]
%   results(27) : sideslip angle of CG [rad]

figure(7)
subplot(2, 2, 1)
plot(tout, results1(11,:)*180/pi, s1, sline, 2); hold on;
plot(tout, results2(11,:)*180/pi, s2, sline, 2); hold on;
title('FL slip angle [deg]');
xlabel('Time[sec]');
ylabel('FL slip angle [deg]');
legend('steer ','brake ');

subplot(2, 2, 2)
plot(tout, results1(12,:)*180/pi, s1, sline, 2); hold on;
plot(tout, results2(12,:)*180/pi, s2, sline, 2); hold on;
title('FR slip angle [deg]');
xlabel('Time[sec]');
ylabel('FR slip angle [deg]');
legend('steer ','brake ');

subplot(2, 2, 3)
plot(tout, results1(13,:)*180/pi, s1, sline, 2); hold on;
plot(tout, results2(13,:)*180/pi, s2, sline, 2); hold on;
title('RL slip angle [deg]');
xlabel('Time[sec]');
ylabel('RL slip angle [deg]');
legend('steer ','brake ');

subplot(2, 2, 4)
plot(tout, results1(14,:)*180/pi, s1, sline, 2); hold on;
plot(tout, results2(14,:)*180/pi, s2, sline, 2); hold on;
title('RR slip angle [deg]');
xlabel('Time[sec]');
ylabel('RR slip angle [deg]');
legend('steer ','brake ');


figure(8)
plot(tout, results1(29,:)*180/pi, s1, sline, 2); hold on;
plot(tout, results2(29,:)*180/pi, s2, sline, 2); hold on;
title('sideslip angle of CG [deg]');
xlabel('Time[sec]');
ylabel('sideslip angle of CG [deg]');
legend('steer ','brake ');


% Plot results(23) : FL lateral tire force [N]
%      results(24) : FR lateral tire force [N]
%      results(25) : RL lateral tire force [N]
%      results(26) : RR lateral tire force [N]
figure(9)
subplot(2, 2, 1)
plot(tout, results1(23,:), s1, sline, 2); hold on;
plot(tout, results2(23,:), s2, sline, 2); hold on;
title('FL lateral tire force [N]');
xlabel('Time[sec]');
ylabel('FL lateral tire force [N]');
legend('steer ','brake ');

subplot(2, 2, 2)
plot(tout, results1(24,:), s1, sline, 2); hold on;
plot(tout, results2(24,:), s2, sline, 2); hold on;
title('FR lateral tire force [N]');
xlabel('Time[sec]');
ylabel('FR lateral tire force [N]');
legend('steer ','brake ');

subplot(2, 2, 3)
plot(tout, results1(25,:), s1, sline, 2); hold on;
plot(tout, results2(25,:), s2, sline, 2); hold on;
title('RL lateral tire force [N]');
xlabel('Time[sec]');
ylabel('RL lateral tire force [N]');
legend('steer ','brake ');

subplot(2, 2, 4)
plot(tout, results1(26,:), s1, sline, 2); hold on;
plot(tout, results2(26,:), s2, sline, 2); hold on;
title('RR lateral tire force [N]');
xlabel('Time[sec]');
ylabel('RR lateral tire force [N]');
legend('steer ','brake ');

% Plot results(30)  : FL brake torque[Nm]
%      results(31)  : FR brake torque[Nm]
%      results(32)  : RL brake torque[Nm]
%      results(33)  : RR brake torque[Nm]

figure(10)

subplot (2, 2, 1)
plot(tout, results1(30,:), s1, sline, 2); hold on;
plot(tout, results2(30,:), s2, sline, 2); hold on;
title('FL brake torque[Nm]');
xlabel('Time[sec]');
ylabel('FL brake torque[Nm]');
legend('steer ','brake ');

subplot (2, 2, 2)
plot(tout, results1(31,:), s1, sline, 2); hold on;
plot(tout, results2(31,:), s2, sline, 2); hold on;
title('FR brake torque[Nm]');
xlabel('Time[sec]');
ylabel('FR brake torque[Nm]');
legend('steer ','brake ');


subplot (2, 2, 3)
plot(tout, results1(32,:), s1, sline, 2); hold on;
plot(tout, results2(32,:), s2, sline, 2); hold on;
title('RL brake torque[Nm]');
xlabel('Time[sec]');
ylabel('RL brake torque[Nm]');
legend('steer ','brake ');


subplot (2, 2, 4)
plot(tout, results1(33,:), s1, sline, 2); hold on;
plot(tout, results2(33,:), s2, sline, 2); hold on;
title('RR brake torque[Nm]');
xlabel('Time[sec]');
ylabel('RR brake torque[Nm]');
legend('steer ','brake ');



 figure(11)

 plot(tout, results1(29,:), s1, sline, 2); hold on;
 plot(tout, results2(29,:), s2, sline, 2); hold on;

 title('Lateral accleration [m/s^2]');
 xlabel('Time[sec]');
 ylabel('Laterall accleration [m/s^2]');
legend('steer ','brake ');




 