clear;
im=imread('obj1__0.png');
number=10;%��ͬ���ݵĸ���
Ynew=[];%���ǻ�ԭ�������
%I=rgb2gray(im);%ת��Ϊ�Ҷ�
I=im;
k=5;%knn��k
Y=double(I);
%################################
%��һ�δ����ǲ�����ͬ���ݵĸ���
%for i=1:number
%   Y(1:600,i)=Y(1:600,1);
%end
%###############################
a = 10;  
b = 4;  
n_gaussian = randn(128, 128);
YY=Y+n_gaussian;
Yg=mapminmax(YY);
a=fkNN(Yg,k);
[m,n]=size(Y);
W=zeros(n,n);
%******************************
%��һ�ε������ǲ����Ƿ���k��������
for i=1:n
    aa=a(i,1:k);
    for j=1:n
        if any(aa==j)
            W(i,j)=1;
            W(j,i)=1;%����������ԳƵ�
        end
    end
    W(i,i)=0;
end
%****************************
%*********************************
%��һ����ʹ�þ����������
% for i=1:m
%    aa=a(i,k+1);
%    for j=1:n
%        if norm(Yg(:,i)-Yg(:,j))<=aa
%            W(i,j)=1;
%            W(j,i)=1;%����������ԳƵ�
%        end
%    end
%    W(i,i)=0
% end
%n_gaussian = a + b .* randn(1200, 1920);  
[Z, E] = sparse_graph_LRR(YY,W,0.1, 5, 8.0, 2.5, 0);
Ynew=Y*Z+E;