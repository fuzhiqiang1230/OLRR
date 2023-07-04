%��������������Խ��ڽ��д�����ϵ��
%time�Ǵ����Ĵ���
%k�Ǽ�Ȩ
function result = sendk(data, time, kn)
[m,~] = size(data);
if nargin<3
    kn = 1;
elseif nargin<2   
    time = m;
end
datanew = data;
dataold = data;
for i = 1:time
    for j = 1:m
        listn = find(datanew(j,:)>0);%�ҵ�����0��ÿһ��
        newsum = sum(data(listn,:));%
        listsum = find(newsum>0);
        listk = setdiff(listsum,listn);%�ҵ����
        if(isempty(listk))
            continue;
        end
        datanew(j,listk) = kn^i;
    end
    if dataold == datanew
        break;
    end
    dataold = datanew;
end
result = dataold;
        
        