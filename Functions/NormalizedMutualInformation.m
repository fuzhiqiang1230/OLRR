function [nmi]=NormalizedMutualInformation(A,IDX,patternNum,k)
index_original=A;%��ԭʼ��Ŵ������
index_new=IDX;%�Ѿ������ı�Ŵ���þ���
k_original=max(index_original);%ԭʼ�����
k_new=max(index_new);%����õ��������
number_original=zeros(k_original,1);%��¼ԭʼ���ݼ�ÿ���������
number_new=zeros(k_new,1);%��¼��������ÿ���������
number_mutual=zeros(k_original,k_new);%��¼ԭʼ���;��������֮��Ľ������д���ԭʼ��������д������������
for i=1:patternNum
    x_temp=index_original(i);%�õ���������ԭʼ����
    y_temp=index_new(i);%�õ��������ľ���������
    number_original(x_temp)=number_original(x_temp)+1;
    number_new(y_temp)=number_new(y_temp)+1;
    number_mutual(x_temp,y_temp)=number_mutual(x_temp,y_temp)+1;
end
sum_up=0;
sum_down_left=0;
sum_down_right=0;
for i=1:k_new
    if number_new(i)>0
        if number_new(i)==patternNum
            sum_down_right=1;
        else
            sum_down_right=sum_down_right+number_new(i)*log(number_new(i)/patternNum);%����NMI��ʽ���ӵ��Ұ벿��
        end
    end
end
for i=1:k_original
    sum_down_left=sum_down_left+number_original(i)*log(number_original(i)/patternNum);%����NMI��ʽ���ӵ���벿��    
    for j=1:k_new
        if number_mutual(i,j)==0
            temp=0;
        else
            temp=number_mutual(i,j)*log((patternNum*number_mutual(i,j))/(number_original(i)*number_new(j)));
        end
        sum_up=sum_up+temp;%����NMI��ʽ�ķ�ĸ
    end
end
nmi=sum_up/sqrt(sum_down_left*sum_down_right);
    
