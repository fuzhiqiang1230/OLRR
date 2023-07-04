%������constractmap��õ������ͼ����ֵ��ת��Ϊ���ڴ���ͼ����ֵ��
%���룺
%data���������ͼ����ֵ��
function [graph] = transmit(data, time)
	[m,~] = size(data);
    if nargin < 2
        time = m;
    end
	datasave = data;
	datan = data;
    for i = 1:time
        for j = 1:m
            nlist = find(datasave(j,:)>0);
            datan(j,:) = datan(j,:) + sum(datasave(nlist(:),:));
        end
        datasave = datan;
    end
for i = 1:m
    for j = 1:m
        if datasave(i,j) >0
            datasave(i,j) = 1;
        end
    end
    datasave(i,i) = 0;
end
graph = datasave;