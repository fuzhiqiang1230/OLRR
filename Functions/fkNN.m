function [R,D] = fkNN(X,k,distanc,target)
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
if distanc == 0
    for i = 1:n
        for j = 1:n
			distanceX(i,j) = norm(saveX(:,i)-saveX(:,j));
        end
    end
    if target == 0
        for i = 1:n
            [a,b] = sort(distanceX(i,:));
            result(i,:) = b(2:k+1);
            dis(i,:) = a(2:k+1);
        end 
    else
        result = zeros(n,k);
        for i = 1:n
            [a,b] = sort(distanceX(i,:),'descend');
            result(i,:)=b(1:k);
            dis(i,:) = a(1:k);
        end
    end
else
    for i = 1:n
		for j = 1:n
			distanceX(i,j) = dot(X(:,i),X(:,j))/(norm(X(:,i))*norm(X(:,j)));
		end
    end
    if target == 0
        for i = 1:n
            [a,b] = sort(distanceX(i,:));
            result(i,:) = b(1:k);
            dis(i,:) = a(1:k);
        end 
    else
        result = zeros(n,k);
        for i = 1:n
            [a,b] = sort(distanceX(i,:),'descend');
            result(i,:)=b(2:k+1);
            dis(i,:) = a(2:k+1);
        end
    end
end
R = result;
D = dis;