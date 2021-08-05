% ******************** Output variables ***************************
% result variable : tout , result(19)
% results(1) : simeulation time [sec]
% results(2) : vehicle longitudinal velocity [m/s]
% results(3) : vehicle lateral velocity [m/s]
% results(4) : x position (global coordinate) [m]
% results(5) : y position (global coordinate) [m]
% results(6) : x position (local coordinate) [m]
% results(7) : y position (local coordinate) [m]
% results(8) : front wheel steer angle [rad]
% results(9) : yaw velocity [rad/sec]
% results(10) : yaw angle [rad]
% results(11) : FL slip angle
% results(12) : FR slip angle
% results(13) : RL slip angle
% results(14) : RR slip angle
% results(15) : FL wheel speed [m/s]
% results(16) : FR wheel speed [m/s]
% results(17) : RL wheel speed [m/s]
% results(18) : RR wheel speed [m/s]                
% results(19) : FL longitudinal tire force [N]
% results(20) : FR longitudinal tire force [N]
% results(21) : RL longitudinal tire force [N]
% results(22) : RR longitudinal tire force [N]
% results(23) : FL lateral tire force [N]
% results(24) : FR lateral tire force [N]
% results(25) : RL lateral tire force [N]
% results(26) : RR lateral tire force [N]
% results(27) : sideslip angle of CG [rad]
% results(28) : longitudinal acceleration [m/s^2]
% results(29) : laterall accleration [m/s^2]
% *****************************************************************
close all;
load('results1.mat');
% load('results2.mat');
% load('results3.mat');
% load('results4.mat');
tout = results1(1,:);

s1 = 'r--';
s2 = 'k-';
s3 = 'b-.';
s4 = 'b:';
s5 = 'g-';
sline = 'LineWidth';

% Plot results1(2) : vehicle longitudinal velocity
%      results1(15:18) : FL, FR, RL, RR wheel velocity
figure(1)
plot(tout, results1(2,:), s1, sline, 1.5); hold on;
plot(tout, results1(15,:), s2, sline, 1.5); hold on;
plot(tout, results1(16,:), s3, sline, 1.5); hold on;
plot(tout, results1(17,:), s4, sline, 1.5); hold on;
plot(tout, results1(18,:), s5, sline, 1.5);
title('Vehicle and wheel speed');
xlabel('Time[sec]');
ylabel('Velocity[m/s]');
legend('vehicle speed', 'FL', 'FR', 'RL', 'RR');

% Plot results1(4) : x position (global coordinate) [m]
%      results1(5) : y position (global coordinate) [m]
figure(2)
t = results1(1,:)';
nt = size(results1(1,:));
nt = nt(2);

plot(results1(4,:), results1(5,:), s1, sline, 1.5);
axis equal;
x = results1(4,:)';
y = results1(5,:)';
psi = results1(10,:)';
    
numcar = 20;  % approximate number of cars in the plot
		
	LX = [1 -1.5 -1.5 1 1; 1.5/2 1.5/2 -1.5/2 -1.5/2 1.5/2]*2;
	ii = 0;
	for i = 1:floor(nt/numcar):nt
       	ii = ii + 1;
        RotYaw=[cos(psi(i)) -sin(psi(i)); sin(psi(i))  cos(psi(i))];
        Xvectortemp=RotYaw*LX*1.0;
	    Xvector=Xvectortemp+[x(i)*ones(1,5); y(i)*ones(1,5)];
	    line(Xvector(1,1:4),Xvector(2,1:4),'Color','b','LineWidth',1);
	    line(Xvector(1,4:5),Xvector(2,4:5),'Color','b','LineWidth',3);
	    timetext=num2str(t(i,1));
	    % if ( fix(ii/3)*3 == ii )
	    %     text(x(i),y(i),timetext);
	    % end
	end;
title('Vehicle trajectory at global coordinate');
xlabel('X-axis[m]');
ylabel('Y-axis[m]');

% Plot results1(8) : front wheel steer angle [rad]
figure(4)
subplot(2,2,1)
plot(tout, results1(8,:)*180/pi, s2, sline, 1.5);
title('Front wheel steer angle');
% xlabel('Time[sec]');
ylabel('Angle[deg]');
% plot slip angle of tire
subplot(2,2,2)
plot(tout, results1(11,:)*180/pi, s1, sline, 1.5); hold on
plot(tout, results1(12,:)*180/pi, s2, sline, 1.5); hold on
plot(tout, results1(13,:)*180/pi, s3, sline, 1.5); hold on
plot(tout, results1(14,:)*180/pi, s4, sline, 1.5);
title('Slip angle of tire');
ylabel('Angle[deg]');
% xlabel('Time[sec]');
legend('FL', 'FR', 'RL', 'RR');
% plot sideslip angle of vehicle CG
subplot(2,2,3)
plot(tout, results1(27,:)*180/pi, s1, sline, 1.5);
title('Sideslip angle of CG');
ylabel('Angle[deg]');
xlabel('Time[sec]');
% plot lateral tire force
subplot(2,2,4)
plot(tout, results1(23,:), s1, sline, 1.5); hold on
plot(tout, results1(24,:), s2, sline, 1.5); hold on
plot(tout, results1(25,:), s3, sline, 1.5); hold on
plot(tout, results1(26,:), s4, sline, 1.5);
title('Lateral tire force');
ylabel('Force[N]');
xlabel('Time[sec]');
legend('FL', 'FR', 'RL', 'RR');

% Plot results1(9) : yaw velocity [rad/sec]
%      results1(10) : yaw angle [rad]
figure(5)
subplot(2,1,1)
plot(tout, results1(10,:)*180/pi, s2, sline, 1.5);
title('Yaw angle of vehicle of CG');
ylabel('Yaw angle[deg]');
subplot(2,1,2)
plot(tout, results1(9,:)*180/pi, s3, sline, 1.5);
title('Yaw rate of vehicle of CG');
ylabel('Yaw rate[deg/sec]');
xlabel('Time[sec]');

% Plot results1(28) : longitudinal acceleration [m/s^2]
%      results1(29) : laterall accleration [m/s^2]
figure(10)
subplot(2,1,1)
plot(tout, results1(28,:)/9.8, s2, sline, 1.5);
title('Longitudinal acceleration');
ylabel('Acceleration[g]');
subplot(2,1,2)
plot(tout, results1(29,:)/9.8, s2, sline, 1.5);
title('Lateral acceleration');
xlabel('Time[sec]');
ylabel('Acceleration[g]');


figure(11)

plot(results1(4,:), results1(5,:), s1, sline, 1.5);