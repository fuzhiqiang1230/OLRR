%��fkNN��������Ľ��ھ���;���ת��Ϊÿһ��Ԫ�غ͵�����Ԫ�صľ������
%���룺
%data1����������ž���
%data2����������ž����Ӧ�ľ������
%max���������ڵ�Ԫ�صľ��룬Ĭ��Ϊ1000
function result = mapdis(data1,data2,max)
if nargin < 3
    max = 1000;
end
[m,n] = size(data1);
savedata = ones(m,m)*max;
for i = 1:m
    savedata(i,i) = 0;
end
for i = 1:m
    for j = 1:n
		savedata(i,data1(i,j)) = data2(i,j);
    end
end
result = savedata;
 