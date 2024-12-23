%=====================================
% Simulation: 
% Aftering transfering NN model in 
% MATLAB format, verify models 
% availbility with a given random point.
%=====================================

clc
close all
clear all


%% theta Parameter settings



theta = 1;
direction = 1;

alpha_x = 0.0527;
alpha_y = 0.0187;
alpha_z = 1.7873;
alpha = [alpha_x,alpha_y,alpha_z];

beta_x = -5.4779;
beta_y = -7.0608;
beta_z = -1.7382;
beta = [beta_x,beta_y,beta_z];

num_compare = 50;
result_p = zeros(1,2,num_compare);
result_u = zeros(1,num_compare);
cnt = 1;

% read A,B,C,D matrices:
A = [0 1 ; 0 -alpha(direction)];
B=[0;beta(direction)];
C = [1 0];
D=0;
G=ss(A,B,C,D);

Gd=c2d(G,0.1);
Ad=Gd.A;
Bd=Gd.B;

x1 = 1.0;
x2 = -1;
r = 0;
Q = [100 0;0 5];  % State cost (penalize deviation of states)
R = 0.1;       % Control effort cost (penalize large control inputs)
% init point and ref for z direction.
if direction == 2
    x1 = 1.55;
    x2 = -0.5;
    R = 0.01;
    Q = [10 0;0 0.1];  % State cost (penalize deviation of states)
    r = 1.5;
end

x_r = [r;0];
% given a random point, go to reference r. with num_compare steps.
p_tt = eye(2);
tic; 
count = 1;
while cnt <= num_compare
    
    % OPtimization part
    if cnt == 1
        x = [x1;x2];
    else
        x = xtt;
    end
 
    [opt_u,opt_p,theta] = multi_agent_algorithm(x,x_r,Ad,Bd,R,Q);
    eig(opt_p)

    % satruate input
    if opt_u > 0.5
        opt_u = 0.5;
    elseif opt_u < -0.5
        opt_u = -0.5;
    end
    xtt = Ad*x +Bd*opt_u

    % save data to compare
    % saving matrix looks like: 
    % ================================
    %   x1  delta_x1 norm(x) u delta_v  
    %   x2  delta_x2   0     0    0
    % ================================
    delta_x = xtt-x;
    delta_v = sqrt(xtt'*opt_p*xtt) - sqrt(x'*p_tt*x);
    norm_xt = norm(x);    
    % opt_result(:,:,cnt) = opt_p;
    opt_result(1,4,cnt) = opt_u;
    opt_result(1:2,1,cnt) = x;
    opt_result(1:2,2,cnt) = delta_x;
    opt_result(1,5,cnt) = delta_v;
    opt_result(1,3,cnt) = norm_xt;

    result_input(cnt,:) = x;

    % next x(t+1)
    % xtt = [x(2);-9.8*sin(x(1))] +[0;1]*opt_u;

    cnt = cnt + 1;
    p_tt = opt_p;

end


elapsed_time = toc;

disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
% writematrix(opt_result, save_name);
%%  display
figure(1)
plot(result_input(:,1),result_input(:,2),'-+')
xlabel('x1 [m]')
ylabel('x2 [m/s]')
grid("on")
title('trajectory with given random points')

figure(2)
plot(1:num_compare,reshape(opt_result(1,5,:),1,num_compare))
xlabel('count')
ylabel('\Delta V')
grid("on")
title('Derivative of Lyapunov Function')