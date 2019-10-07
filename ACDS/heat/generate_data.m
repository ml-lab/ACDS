
%------------------------------------------------------------------------
%--- Heat Equation in two dimensions-------------------------------------
%--- Solves Ut=alpha*(Uxx+Uyy)-------------------------------------------
%------------------------------------------------------------------------
clc;
close all;
clear all;
%--dimensions...........................................................
N = 100;  
DX=0.1; % step size
DY=0.1;
Nx=10; 
Ny=10;
X=0:DX:Nx;
Y=0:DY:Ny;
alpha=100; % arbitrary thermal diffusivity 
%--boundary conditions----------------------------------------------------
%U(1:N+1,1:N+1) = 0 ; 
%U(1,1:N+1) = 0;
%U(N+1,1:N+1) = 0;  
%U(1:N+1,1) = 0;  
%U(1:N+1,N+1) = 0;  
%--initial condition------------------------------------------------------
%U(29:31,29:31)=300; % a heated patch at the center
%U(15:17,15:17)=300;
%U(42:44,42:44)=300;
%U(15:17,42:44)=300;
%U(42:44,15:17)=300;

%U(15:18,29:31)=300; % a heated patch at the center
%U(15:17,15:17)=300;
%U(42:44,42:44)=300;
%U(15:17,42:44)=300;
%U(42:45,29:31)=300;
[X1,X2] = meshgrid(X,Y);
XX = [X1(:) X2(:)];
mu_1 = [3 5];
sigma_1 = [0.25 0.3; 0.3 1];
mu_2 = [7 5];
sigma_2 = [0.25 0.3; 0.3 1];
U = mvnpdf(XX,mu_1,sigma_1)+mvnpdf(XX,mu_2,sigma_2);
U = reshape(U,length(X),length(Y));
Umax=max(max(U));

surf(U);
%-------------------------------------------------------------------------
DT = DX^2/(2*alpha); % time step 
M=600; % maximum number of allowed iteration
%---finite difference scheme----------------------------------------------
fram=0;
Ncount=0;
loop=1;
n = 0;
while loop==1;
    n=n+1;
   ERR=0; 
   U_old = U;
for i = 2:N
for j = 2:N
   Residue=(DT*((U_old(i+1,j)-2*U_old(i,j)+U_old(i-1,j))/DX^2 ... 
                      + (U_old(i,j+1)-2*U_old(i,j)+U_old(i,j-1))/DY^2) ...
                      + U_old(i,j))-U(i,j);
   ERR=ERR+abs(Residue);
  U(i,j)=U(i,j)+Residue;
end
end
if(ERR>=0.01*Umax)  % allowed error limit is 1% of maximum temperature
    Ncount=Ncount+1;
         if (mod(Ncount,50)==0) % displays movie frame every 50 time steps
              fram=fram+1;
              surf(U);
              axis([1 N+1 1 N+1 ])
              h=gca; 
              get(h,'FontSize') 
              set(h,'FontSize',12)
              colorbar('location','eastoutside','fontsize',12);
              xlabel('X','fontSize',12);
              ylabel('Y','fontSize',12);
              title('Heat Diffusion','fontsize',12);
              fh = figure(1);
             set(fh, 'color', 'white'); 
            F=getframe;
         end
 
 %--if solution do not converge in 2000 time steps------------------------
 
    if(Ncount>M)
        loop=0;
        disp(['solution do not reach steady state in ',num2str(M),...
            'time steps'])
    end
    
 %--if solution converges within 2000 time steps..........................   
    
else
    loop=0;
    disp(['solution reaches steady state in ',num2str(Ncount) ,'time steps'])
end
T(:,:,n) = U;
end
%------------------------------------------------------------------------
%--display a movie of heat diffusion------------------------------------
 %movie(F,fram,1)
%------END---------------------------------------------------------------
x = X;
y = Y;
dt = DT;
dx = DX;
dy = DY;
save('heat','T','x','y','dt','dx','dy')
