function c_pseudo=goldseq(n_slot,l)
%用于生成PRS序列的伪随机序列c，n_slot为slot number，l是序列映射到的slot内的OFDM符号
N_c= 1600 ;
Mpn= 5000;
c_pseudo=zeros(1,Mpn);
n_ID=1;
N_sym=14;
x10 = 1;
x11 =zeros(1,30);
x1 = [x10 x11];
for i= 1:(N_c+Mpn -31)
    x1(i+31) =mod((x1(i+3)+x1(i)),2);
end
cinit =mod((2^22*floor(n_ID/1024)+2^10*(N_sym*n_slot+l+1)*(2*mod(n_ID,1024)+1)+mod(n_ID,1024)),2^31);%初始序列生成
x21 =de2bi(cinit);
len =length(x21);
x22 =zeros(1,31-len);
x2  =[x21 x22];

for i=1:(N_c +Mpn -31)
    x2(i+31) = mod((x2(i+3)+x2(i+1)+x2(i)),2);
end

for i =1:Mpn
    c_pseudo(i)= mod((x1(i+N_c)+x2(i+N_c)),2);
end
end
