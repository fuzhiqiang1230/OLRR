%���������Ҫ���ڽ����������ת��Ϊͼ��ϵ
%���룺fkNN�Ľ��
%������������Լ�ÿһ�����Լ�����ڵľ�Ϊ1�����ǵ�Ϊ0
function [result] = constractmap(data)
	[m,n] = size(data);
	savedata = zeros(m,m);
	for i = 1:m
        savedata(i,data(i,:)) = 1;
	end
	result = savedata;