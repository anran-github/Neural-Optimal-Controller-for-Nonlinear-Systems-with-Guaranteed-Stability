function [u_ii, p_ii, theta_ii] = drone_original_optimation(x,r, Ad, Bd,R,Q)
     
    x_r = r;
    
    % variables to optimize
    u = sdpvar(1,1,'full');
    P=sdpvar(2,2,'symmetric');
    theta=sdpvar(1,1,'full');

    
    Objective = R*(norm(u))^2+(Ad*x+Bd*u-x_r)'*Q*(Ad*x+Bd*u-x_r)+((x-x_r)'*P*(x-x_r))+exp(-theta);
    Constraints = [P>=1e-10;theta>=0;
    ((x-x_r)'*P*(x-x_r))>=((1.5*theta)^2)*(x-x_r)'*(x-x_r);
    ((Ad*x+Bd*u-x_r)'*P*(Ad*x+Bd*u-x_r))<=((0.5*theta)^2)*(x-x_r)'*(x-x_r)];
    
    % opt=sdpsettings('solver','bmibnb');
    opt=sdpsettings('solver','fmincon','MaxIter',2000);
    sol=optimize(Constraints,0.000001*Objective,opt)
    u_ii = double(u);
    p_ii = double(P);
    theta_ii=double(theta);

end