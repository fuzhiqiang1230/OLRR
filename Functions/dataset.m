r = 10; %�ھ�
w = 6;  %   �⾶= r+w
num = 2000; %���ݵ�����
RandStream.setDefaultStream ...
     (RandStream('mt19937ar','seed',sum(100*clock))); %�Ƽ������
dR = rand(1,num)*w;
dThet = rand(1,num)*2*pi;
X = (r+dR) .* cos(dThet);
Y = (r+dR).* sin(dThet);
%ԭʼ����
data = [X;Y];
dataTargets= myHardlim(Y); %Ŀ����� logical ��
dataTargets = dataTargets .* 1.0;% logical �� ת��Ϊ double��
flag1 = find(dataTargets== 1) ;
plot(data(1,flag1),data(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data(1,flag0),data(2,flag0),'b.','MarkerSize',5);
title('ԭʼ����')
%�ƶ�������� data1
dv= 1;  % ���������¡��Ĵ�ֱ���� ��������ʾ��ø�Զ
dh = 10 ;   %���������¡���ˮƽ���� ��������ʾ���� ��ø�Զ
X = data(1,:);
Y = data(2,:);
flag = find( Y < 0 );
Y(1,flag) = Y(1,flag) - dv;
X(1,flag) = X(1,flag) + dh;
data1 = [X;Y];
figure;
flag1 = find(dataTargets== 1) ;
plot(data1(1,flag1),data1(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data1(1,flag0),data1(2,flag0),'b.','MarkerSize',5);
title('�ƶ� dv=1,dh=10 ������� ')
%�ƶ�������� data2
dv= -4;  % ���������¡��Ĵ�ֱ���� ��������ʾ��ø�Զ
dh = 10 ;   %���������¡���ˮƽ���� ��������ʾ���� ��ø�Զ
X = data(1,:);
Y = data(2,:);
flag = find( Y < 0 );
Y(1,flag) = Y(1,flag) - dv;
X(1,flag) = X(1,flag) + dh;
data2 = [X;Y];
figure;
flag1 = find(dataTargets== 1) ;
plot(data2(1,flag1),data2(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data2(1,flag0),data2(2,flag0),'b.','MarkerSize',5);
title('�ƶ� dv=-4,dh=10 ������� ');
%�ƶ�������� data3
dv= -5;  % ���������¡��Ĵ�ֱ���� ��������ʾ��ø�Զ
dh = 10 ;   %���������¡���ˮƽ���� ��������ʾ���� ��ø�Զ
X = data(1,:);
Y = data(2,:);
flag = find( Y < 0 );
Y(1,flag) = Y(1,flag) - dv;
X(1,flag) = X(1,flag) + dh;
data3 = [X;Y];
figure;
flag1 = find(dataTargets== 1) ;
plot(data3(1,flag1),data3(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data3(1,flag0),data3(2,flag0),'b.','MarkerSize',5);
title('�ƶ� dv=-5,dh=10 ������� ');
%�ƶ�������� data4
dv= -6;  % ���������¡��Ĵ�ֱ���� ��������ʾ��ø�Զ
dh = 10 ;   %���������¡���ˮƽ���� ��������ʾ���� ��ø�Զ
X = data(1,:);
Y = data(2,:);
flag = find( Y < 0 );
Y(1,flag) = Y(1,flag) - dv;
X(1,flag) = X(1,flag) + dh;
data4 = [X;Y];
figure;
flag1 = find(dataTargets== 1) ;
plot(data4(1,flag1),data4(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data4(1,flag0),data4(2,flag0),'b.','MarkerSize',5);
title('�ƶ� dv=-6,dh=10 ������� ');
%�������ݼ����ļ�
currentpath = eval( 'pwd' ); %��ȡ��ǰ����Ŀ¼·��
disp('current path : ');
disp(currentpath);
ts = strcat(currentpath,'\shuangyuedata.mat');
save(ts,'data','data1','data2','data3','data4','dataTargets');