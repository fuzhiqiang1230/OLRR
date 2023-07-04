function U=NJW(data,k)
%%NJW�㷨  ѡȡ���Ͼ����ǰK���������ֵ��Ӧ������������ʹ����R��k���ռ��й�����ԭ����һһ��Ӧ�ı��������ڸÿռ��ڽ��о���
%%data n*m n��������  m ά��
%% k  ѡ��ǰk���������ֵ��Ӧ����������

%�������ƾ���
affinity = AffinityMatrix(data);
% ����ԽǾ���
D=eye(size(affinity,1));
for i=1:size(affinity,1)
    D(i,i) = sum(affinity(i,:));
end

%����������˹���󣬲��÷ǹ淶������
% L=D-affinity;

% �淶��
for i=1:size(affinity,1)
    for j=1:size(affinity,2)
        L(i,j) = affinity(i,j) / (sqrt(D(i,i)) * sqrt(D(j,j)));  
    end
end


%��������ֵ��������
[eigVectors,eigValues] = eig(L);

% ѡȡǰK���������ֵ
[eigValues, ind] = sort(diag(eigValues), 'descend');

nEigVec =eigVectors(:,ind(1:k));

% �����һ������U�ӻ�õ���������
U=zeros(size(nEigVec,1),k);
for i=1:size(nEigVec,1)
    n = sqrt(sum(nEigVec(i,:).^2)); 
    if n == 0
         U(i,:) = nEigVec(i,:);
    else
        U(i,:) = nEigVec(i,:) ./ n;
    end
end
U = kmeans(U,k);
end

