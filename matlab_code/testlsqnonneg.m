clear all;clc;
load Salinas.mat;
load Endmembers4.mat;
HSI=reshape(salinas,[],size(salinas,3));
num=size(HSI,1);
[Dim,endmember_num]=size(endmebmers4);
test_num=num;
x=zeros(endmember_num,test_num);
tic
for i=1:test_num
    x(:,i)=lsqnonneg(endmebmers4,HSI(i,:)');
    %x(:,i)=lsqnonneg_ori(endmembers30,HSI(i,:)');
 end
toc
tic
for i=1:test_num
    %x(:,i)=lsqnonneg(endmembers3,HSI(i,:)');
    x(:,i)=lsqnonneg_ori(endmebmers4,HSI(i,:)');
 end
toc
tic
for i=1:test_num
    %x(:,i)=lsqnonneg(endmembers3,HSI(i,:)');
    x(:,i)=fast_NNLS(endmebmers4,HSI(i,:)');
 end
toc
% tic;
% x=hyperFcls(HSI',endmembers24);
% toc
% tic;
% x=hyperNnls(HSI',endmembers24);
% toc