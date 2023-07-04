%the  code of One-step Low-Rank Representation based Clustering 
% X---given dataset
% Y---Fuzzy clustering matrix(denoted by F in paper)
% k---the number of class
% lambda1, lambda2 and lambda3 are the parameters
% Iter_max is the max iteration time and is usually setted as 100
function [Z,E,Y,YG,S] = OLRR(X,Z_ini,Y_ini,k,lambda1,lambda2,lambda3,Iter_max)
    %% initialization
    [d,n] = size(X);
    Y = zeros(n,k);
    C1 = zeros(size(X));
    C2 = zeros(size(X));
    C3 = zeros(size(Z_ini));
    mu = 0.01;
    rho = 1.2;
    max_mu = 1e8;
    tol  = 1e-5;
    %% main loop
    for iter = 1:Iter_max
        if iter == 1
            Z = Z_ini;
            Y = Y_ini;
            S = 1;
            J = Z;
            U = X;
            clear Z_ini
        end
       %% update E
        temp1 = X-U+C1/mu;
        temp2 = lambda3/mu;
        E = max(0,temp1-temp2) + min(0,temp1+temp2);   
       %% update Z
       Z = pinv((lambda2+mu)*eye(n)+mu*U'*U)*(lambda2*(Y*Y'-eye(n))+mu*(U'*(U+C2/mu)+J-C3/mu));
%         Z = pinv((lambda2+mu)*eye(n)+mu*XTX)*(lambda2*(Y*S*Y')+mu*(X'*(X-E+C1/mu)+J-C3/mu));
        Z = Z - diag(diag(Z));
        Z = max(Z,0);
       %% update J
        [AU,SU,VU] = svd(Z+C3/mu,'econ');
        AU(isnan(AU)) = 0;
        VU(isnan(VU)) = 0;
        SU(isnan(SU)) = 0;
        SU = diag(SU);    
        SVP = length(find(SU>lambda1/mu));
        if SVP >= 1
            SU = SU(1:SVP)-lambda1/mu;
        else
            SVP = 1;
            SU = 0;
        end
        J = AU(:,1:SVP)*diag(SU)*VU(:,1:SVP)'; 
       %% update U
        YTY = Y*Y';
        LY = diag(sum(YTY))-YTY;
        U = mu*(X-E+C1/mu-C2/mu*(eye(n)-Z'))*pinv(4*LY+mu*eye(n)+mu*(eye(n)-Z)*(eye(n)-Z'));
        D =  L2_distance_1(U,U);
       %% update Y S
       [Y,S] = updateYS(D,Z,S,Y,k,lambda2,n);
       %% update others
       YG = YTY;
        L1 = X-U-E;
        L2 = U-U*Z;
        L3 = Z-J;
        C1 = C1+mu*L1;
        C2 = C2+mu*L2;
        C3 = C3+mu*L3;
        mu = min(rho*mu,max_mu);
        %% 
        leq1 = [max(max(abs(L1))),max(max(abs(L2))),max(max(abs(L2)))];%,max(max(abs(L3)))
        stopC = max(leq1);
        if stopC < tol
            iter
            break;
        end   
    end

    %% 0-1
    function [Y,S] = updateYS(D,Z,S,Y,k,lambda3,n)
        miu = 0.01;
        C = zeros(size(Y));
        for iiter = 1:100
            if iiter == 1
                G = Y;
            end
           %% update S 先改成s
%             S = trace(Z*(G*Y'-eye(n)))/trace((G*Y'-eye(n))*(G*Y'-eye(n))');
%             if S>1
                S=1;
%             end
           %% update Y
            ZI = Z;
            Y = (-D*G+lambda3*ZI*G*S+miu*(G-C/miu))*pinv(lambda3*S*G'*G*S+miu*eye(k));
            for iic = 1:n
                iidx    = 1:k;
                Y(iic,iidx) = EProjSimplex_new(Y(iic,iidx));          % 
            end
           %% update G
            G = (-D*Y+lambda3*ZI'*Y*S+miu*(Y+C/miu))*pinv(lambda3*S*Y'*Y*S+miu*eye(k));
           %% update others
            L = Y-G;
            C = C+miu*L;
            miu = min(1.2*miu,1000000);
           %% 
            leq = [max(max(abs(L)))];
            stop = max(leq);
            if stop < 1e-5
                break;
            end   
        end
    end
end