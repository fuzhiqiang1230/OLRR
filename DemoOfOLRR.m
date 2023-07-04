%This is a demo to show our method.
%One obains the ACC and purity by running this code.
%More details will be open on the GitHub after the review.
clear
clc
dataset_name = 'solar_uni';
load(strcat(dataset_name,'.mat'))
max_iter =100;
fea = X;
gnd = Y;
fea = fea';
fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
n = length(gnd);
nnClass = length(unique(gnd));
kk = nnClass;
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';
Z = constructW(fea',options);
Z = full(Z);
Z1 = Z-diag(diag(Z));         
Z = (Z1+Z1')/2;
DZ= diag(sum(Z));
LZ = DZ - Z;
[F_ini, ~, evs]=eig1(LZ, kk, 0);
Z_ini = Z;
X = fea;
Y_ini = zeros(n,kk);
lambda1 = 10^(3);
lambda2 = 10^(-1);
lambda3 = 10^(2);
[Z,E,Y,YG,S] = OLRR(fea,Z_ini,Y_ini,kk,lambda1,lambda2,lambda3,max_iter);

%% test Y
[~,result_labely] = sort(Y,2,'descend');
resulty = ClusteringMeasure(gnd, result_labely(:,1));
acc = resulty(1);
purity = resulty(3);
