function [rcosw]=rcoswindow(beta, Ts)
%定义升余弦窗，其中beta为滚降系数=循环后缀比率，Ts为包含循环前缀的OFDM符号的长度,Ts为正偶数

t=0:(1+beta)*Ts;
rcosw=zeros(1,(1+beta)*Ts);%beta*Ts=beta*（IFFT_length+GI）=循环后缀个数
%前部滚降
for i=1:beta*Ts

%     rcosw(i)=0.5+0.5*cos(pi- t(i)*pi/(beta*Ts));
    rcosw(i)=0.5+0.5*cos((beta*Ts-t(i))*pi/(beta*Ts));
%     rcosw(i)=0.5+0.5*cos(pi+ t(i)*pi/(beta*Ts));
%     rcosw(i)=0.5+0.5*cos(beta*Ts+(t(i))*pi/(beta*Ts));
end
%中部全1
rcosw(beta*Ts+1:Ts)=1;
%尾部滚降
for j=Ts+1:(1+beta)*Ts+1
    rcosw(j-1)=0.5+0.5*cos((t(j)-Ts)*pi/(beta*Ts));
end
rcosw=rcosw';%变换为列向量
 
    