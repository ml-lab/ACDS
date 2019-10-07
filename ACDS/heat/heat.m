clear all, close all
clc
addpath(genpath(pwd))

%%
load heat;
[xx,yy]=meshgrid(x,y);
index = 50;
eps = 0.2;

%% prepare derivative observations
q = 2:99;
xx = xx(q,q);
yy = yy(q,q);

U = T(q,q,index-1);
surf(xx,yy,U);
dUdt1 = (T(q,q,index) - T(q,q,index-2))/(2*dt);
dUdx = (T(q+1,q,index-1)-T(q-1,q,index-1))/(2*dx);
dUdy = (T(q,q+1,index-1)-T(q,q-1,index-1))/(2*dy);
d2Udx2 = (T(q+1,q,index-1)-2*T(q,q,index-1)+T(q-1,q,index-1))/(dx^2);
d2Udy2 = (T(q,q+1,index-1)-2*T(q,q,index-1)+T(q,q-1,index-1))/(dy^2);
d2Udxdy = (T(q+1,q+1,index-1) + T(q-1,q-1,index-1) ...
          - T(q+1,q-1,index-1) - T(q-1,q+1,index-1))/(4*dx*dy);

xx = xx(:); yy = yy(:); X = [xx yy];
U = U(:); dUdt1 = dUdt1(:); dUdx = dUdx(:); dUdy = dUdy(:);
d2Udx2 = d2Udx2(:); d2Udy2 = d2Udy2(:); d2Udxdy = d2Udxdy(:);

%% pool data
[Theta_true] = pool_data(xx,U,dUdx,dUdy,d2Udx2,d2Udy2,d2Udxdy);
N_star = size(U,1);
N0 = 20;   %sample size
n_s = 20; 
ns_max = 35;
%% prepare true value
[m_x,m_y] = size(Theta_true);
Xi_true = zeros(m_y,1);
Xi_true(11,1) = 1;
Xi_true(12,1) = 1;
Xi_log = zeros(m_y+1,1);
Xi_log(11,1) = 1;
Xi_log(12,1) = 1;

%% record the comparison criteria
chosen_col_last = ones(m_y+1,2);
time = 50;
error = zeros(time,1);
error_l0 = zeros(time,1);
sample = zeros(time,1);

for times =1:time
    dUdt = dUdt1 + eps*randn(size(dUdt1));
    s = 0;
    chosen_index = randsample(N_star, N0);
    x0 = X(chosen_index,:);
    u0 = U(chosen_index,:);
    u0mean = mean(u0);
    u0 = u0-u0mean;
    
    
    %% sequential design
    while (1)
        s = s+1;
        %% GP
        [mu,mu_1,mu_2,sigma] = gp_new(x0,u0);
        u1 = mu(X) + u0mean;
        sigma = sigma/(std(u0)^2);
        du1 = mu_1(X);
        dudy = du1(:,1);
        dudx = du1(:,2);
        d2u1 = mu_2(X);
        d2udy2 = d2u1(1:N_star,1);
        d2udx2 = d2u1(N_star+1:2*N_star,2);
        d2udxdy = d2u1(1:N_star,2);
      
  
        %% pool Data
        [Theta] = pool_data(xx,u1,dudx,dudy,d2udx2,d2udy2,d2udxdy);
        
        %% sparse regression
        Theta_chosen = Theta_true(chosen_index,:);
        eta = dUdt(chosen_index,:);

        mdl = stepwiselm(Theta_chosen,eta,'Criterion','bic');
        chosen_col_1 = mdl.Formula.InModel;
        Theta1 = Theta_chosen(:,chosen_col_1);
        mdl = fitlm(Theta1,eta);
        tol = mdl.RMSE;
        tol = tol/std(eta);
        cof = table2array(mdl.Coefficients(2:end,1));

        chosen_col = double(chosen_col_1)' ;
        error_1 = calError(chosen_col,chosen_col_last);
        
        
        if error_1 == 0
            error_2 = norm(cof - cof_last,2)/norm(cof_last,2);
           
            if error_2 < .01
                break
            end
        end
        
        %% max sample size
        if s > ns_max
            break
        end

         chosen_col_last = chosen_col;
         cof_last = cof;
        
        
        %% optimal design
        [chosen_index]=optimal_design(Theta_true,Theta,chosen_index,n_s,sigma,tol,X);
        x0 = X(chosen_index,:);
        u0 = U(chosen_index,:);
        u0mean = mean(u0);
        u0 = u0-u0mean;
   
    end   

    z1 = zeros(m_y,1);
    z1(chosen_col_1) = cof;
    Xi = z1;
    error(times,1) = calError(chosen_col,Xi_log);
    error_l0(times,1) = norm(Xi-Xi_true,2);
    sample(times,1) = size(chosen_index,1);
  
    
 end

dlmwrite('heat2.txt',error);
dlmwrite('heat2.txt',error_l0,'-append');
dlmwrite('heat2.txt',sample,'-append');

dlmwrite('data2.txt',mean(error));
dlmwrite('data2.txt',std(error),'-append');
dlmwrite('data2.txt',mean(error_l0),'-append');
dlmwrite('data2.txt',std(error_l0),'-append');
dlmwrite('data2.txt',mean(sample),'-append');
dlmwrite('data2.txt',std(sample),'-append');


