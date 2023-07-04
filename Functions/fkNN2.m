function [R,D] = fkNN2(X,k,distanc,target)
%���������Ҫ�����������ÿһ����һ��Ԫ��
%���������
%X--ԭʼ�����ݾ���
%k--�����ĸ�����Ĭ��Ϊ2
%distanc--��ָʹ�õľ��룬0Ϊŷʽ���룬1Ϊ���Ҿ��룬Ĭ��Ϊ0
%target--ѡ����������С��0Ϊ��С1Ϊ���Ĭ��Ϊ0
%�����
if nargin < 4
    target = 0;
end
if nargin < 3
    distanc = 0;
end
if nargin < 2
    k = 2;
end
saveX = X;
[m,n] = size(saveX);
distanceX = zeros(n,n);
result = zeros(n,k);
dis = zeros(n,k);
for i =1:n
    saveX = repmat(X(:,i),1,n);
    saveX = saveX-X;
    saveX = saveX.^2;
    new = sum(saveX);
    distanceX(i,:) = new;
end
[a,b] = sort(distanceX,2);
result = b(:,2:k+1);
dis = a(:,2:k+1);
R = result;
D = dis;