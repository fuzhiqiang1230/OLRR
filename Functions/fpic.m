number=72*4;%ͼƬ������
filepath='.\pic\';%ͼƬ·��
Z=[];%�任����
E=[];%��������
data=[];%ԭʼ����
Y=[];%�任ǰ����
Ynew=[];%��ԭ�������
namelist=[];%ͼƬ������Ŀǰ�ǰ���˳�����е�
k=8;%kNN��k
file_path =  '.\pic\';% ͼ���ļ���·��  
img_path_list = dir(strcat(file_path,'*.bmp'));%��ȡ���ļ���������jpg��ʽ��ͼ��  
img_num = length(img_path_list);%��ȡͼ��������  
if img_num > 0 %������������ͼ��  
        for j = 1:img_num %��һ��ȡͼ��  
            image_name = img_path_list(j).name;% ͼ����  
            image = imread(strcat(file_path,image_name));
            [m,n]=size(image);
            form=reshape(image,m*n,1);
            data=[data,form];
        end
end
% for i=1:number
%     if i<=72
%         filename=strcat('obj','1','__',num2str(i-1),'.png');%obj1__0.png
%     elseif i<=144
%         filename=strcat('obj','9','__',num2str(i-73),'.png');%
%     elseif i<=144+72
%          filename=strcat('obj','16','__',num2str(i-73-72),'.png');%
%     else
%          filename=strcat('obj','2','__',num2str(i-73-72-72),'.png');%
%     end
%     %namelist=[namelist,filename];
%     image=imread(strcat(filepath,filename));
%     [m,n]=size(image);
%     form=reshape(image,m*n,1);
%     data=[data,form];
% end
Y=double(data);
Yg=mapminmax(Y);%��һ��
a=fkNN(Yg,k);
[m,n]=size(Y);
W=zeros(n,n);
%��һ�ε������ǲ����Ƿ���k��������
for i=1:n
   aa=a(i,1:k);
   %aa(1)=0;%�ѽ��ھ�������ĵ�һ��Ԫ����0��Ҳ����û��Ԫ�ر���
   for j=1:n
       if any(aa==j)
           W(i,j)=1;
           W(j,i)=1;
       end
   end
   W(i,i)=0;
end
%****************************
%*********************************
%��һ�����Ǿ�������
% for i=1:n
%     aa=a(i,k+1);
%     for j=1:n
%         if norm(Yg(:,i)-Yg(:,j))<=aa
%             W(i,j)=1;
%             W(j,i)=1;
%         end
%     end
% end
[Z, E] = sparse_graph_LRR(Y,W);%, 0.1, 5, 8.0, 2.5, 0);
Ynew=Y*Z+E;