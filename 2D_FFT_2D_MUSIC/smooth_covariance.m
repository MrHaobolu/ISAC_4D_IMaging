function Rxx=smooth_covariance(Rxxm,K) % 参数1待定协方差矩阵，参数2子阵元数目 
% 用于空间平滑Music算法协方差矩阵获取
[M,MM]=size(Rxxm); % 返回Rxxm矩阵的维度大小
N=M-K+1; % 子阵列的数目
Rxx=zeros(K,K); % 初始化矩阵Rxx，类似于初始化为0
for i=1:N % 第1个子阵列到第N个子阵列
    Rxx=Rxx+Rxxm(i:i+K-1,i:i+K-1); % 向前滑动，注意每个子阵列是k个阵元，不是k-1个
end
Rxx=Rxx/N; % 子阵列协方差矩阵可以相加后平均取代原来意义上的协方差矩阵
end