function result = newtrans(data,k)
%���������Լ�����ڸ���
[m,~] = size(data);
dataold  = data;
datanew = zeros(m,m);
I = eye(m);
for i = 1: 2
    if dataold == datanew
        break;
    end 
    if i ~= 1
        dataold = datanew;
    end
    datanew = dataold*dataold + dataold - k^i^i*I;
    datanew(find(datanew>0)) = 1;
    datanew(find(datanew<0)) = 0; 
end
result = datanew;
    
    