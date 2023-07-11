%% 
clear;
close all;
carrier_count=200;%子载波数
symbols_per_carrier=12;%OFDM符号数

bits_per_symbol=2;%每个调制符号的比特数（4QAM对应的就是2）
% bits_per_symbol=3;%每个调制符号的比特数（8QAM对应的就是3）
% bits_per_symbol=4;%每个调制符号的比特数（16QAM对应的就是4）
% bits_per_symbol=5;%每个调制符号的比特数（32QAM对应的就是5）
% bits_per_symbol=6;%每个调制符号的比特数（64QAM对应的就是6）

IFFT_bin_length=512;%FFT点数
PrefixRatio=1/4;%循环前缀比率 1/6~1/4
GI=PrefixRatio*IFFT_bin_length ;%循环前缀的长度
beta=1/32;%循环后缀比率
GIP=beta*(IFFT_bin_length+GI);%循环后缀长度
SNR=20; %信噪比，单位是dB

%% 基本参数设置

c = 3*10^8;  % 电磁波传播速度， 单位m/s
delta_f = 240*10^3;  % 载波间隔，单位hz
f_c = 24*10^9; % 信号中心频偏, 单位hz

%% ================发送数据生成===================================
baseband_out_length = carrier_count * symbols_per_carrier * bits_per_symbol;%发送比特长度  200*12*4
carriers = (1:carrier_count) + (floor(IFFT_bin_length/4) - floor(carrier_count/2));%-floor(carrier_count/2)将子载波中点移到0点，+floor(IFFT_bin_length/4)将子载波数据搬到512前半部分的中心
conjugate_carriers = IFFT_bin_length - carriers + 2;
%% ***************模拟随机通信信息***************
% baseband_out=zeros(1,baseband_out_length);%纯0序列

%% ***************模拟随机通信信息***************
% baseband_out=ones(1,baseband_out_length);%纯1序列

%% ***************模拟随机通信信息***************
rand( 'twister',0); %随机种子，可以让每次生成的随机数有规律，有了这句话每次生成的随机数具有一样的规律
baseband_out=round(rand(1,baseband_out_length));%生成随机数（模拟随机的通信数据）
%% ***************m序列***************
%==================================================
Tx_matric = zeros(carrier_count*bits_per_symbol,symbols_per_carrier);
for i = 1:symbols_per_carrier
    Order_number=10; %m序列的阶数等于10，m序列长度为2^10 - 1
    mg = zeros(carrier_count*bits_per_symbol,1);
%    mg(1:carrier_count*bits_per_symbol-1,1)  =idinput((2^(Order_number)-1),'prbs')';%采用idinput函数可以生成多种序列
%    生成m序列本源多项式的系数，可以用primpoly(Order_number,'all')得到；
    tmp = primpoly(Order_number,'all'); %生成所有可行的m序列的本源多项式系数，阶数越大，数越多(常数项系数不考虑)
    cur_tmp = int32(tmp(1)); %选择第一个
    % 十进制化为二进制
    f = zeros(1,Order_number+2);
    for j = 1:Order_number+1
        if mod(cur_tmp,2) == 1
            f(j) = 1;
        end
        cur_tmp = idivide(int32(cur_tmp),int32(2),'floor');
    end
    f = f(1,2:Order_number+1);
    tmp = m_generate(f);
    mg(1:carrier_count*bits_per_symbol-1,1) = tmp(1:carrier_count*bits_per_symbol-1);
    Tx_matric(:,i) = mg;
end
baseband_out = reshape(Tx_matric,1,baseband_out_length); 


%% ==============4QAM调制====================================

complex_carrier_matrix=qam4(baseband_out);
 
complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
 
figure(1);
plot(complex_carrier_matrix,'*r');
title('star map of Tx_data');
axis([-2, 2, -2, 2]);
grid on

% %% ==============8QAM调制====================================
% 
% complex_carrier_matrix=qam8(baseband_out);
%  
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-2, 2, -2, 2]);
% grid on


% %% ==============16QAM调制====================================
% 
% complex_carrier_matrix=qam16(baseband_out);
%  
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-4, 4, -4, 4]);
% grid on

% %% ==============32QAM调制====================================
% 
% complex_carrier_matrix=qam32(baseband_out);
%  
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-8, 8, -8, 8]);
% grid on

% %% ==============64QAM调制====================================
% 
% complex_carrier_matrix=qam64(baseband_out);
%  
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-8, 8, -8, 8]);
% grid on


%% =================IFFT===========================
IFFT_modulation=zeros(symbols_per_carrier,IFFT_bin_length);
%通过索引将信息赋值至ofdm符号相应子载波上
IFFT_modulation(:,carriers ) = complex_carrier_matrix ;
IFFT_modulation(:,conjugate_carriers ) = conj(complex_carrier_matrix);
%========================================================
figure(2);
stem(0:IFFT_bin_length-1, abs(IFFT_modulation(2,1:IFFT_bin_length)),'b*-')
grid on
axis ([0 IFFT_bin_length -0.5 4.5]);
ylabel('Magnitude');
xlabel('IFFT Bin');
title('OFDM Carrier Frequency Magnitude');
 
figure(3);
plot(0:IFFT_bin_length-1, (180/pi)*angle(IFFT_modulation(2,1:IFFT_bin_length)), 'go')
hold on
ttt=0:carriers-1;
stem(0:carriers-1, (180/pi)*angle(IFFT_modulation(2,1:carriers)),'b*-');
stem(0:conjugate_carriers-1, (180/pi)*angle(IFFT_modulation(2,1:conjugate_carriers)),'b*-');
axis ([0 IFFT_bin_length -200 +200])
grid on
ylabel('Phase (degrees)')
xlabel('IFFT Bin')
title('OFDM Carrier Phase')
%=================================================================

 
signal_after_IFFT=ifft(IFFT_modulation,IFFT_bin_length,2);%ifft
time_wave_matrix =signal_after_IFFT;
figure(4);
subplot(3,1,1);
plot(0:IFFT_bin_length-1,time_wave_matrix(2,:));
axis([0, 700, -0.2, 0.2]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal, One Symbol Period');
 
%% =====================加循环前缀CP和后缀====================================
XX=zeros(symbols_per_carrier,IFFT_bin_length+GI+GIP);%GI=128,GIP=20
for k=1:symbols_per_carrier    %12
    % for i=1:IFFT_bin_length         %512
    %     XX(k,i+GI)=signal_after_IFFT(k,i);%129--640
    % end
    XX(k,GI+1:GI+IFFT_bin_length)=signal_after_IFFT(k,:);
    % for i=1:GI %1--128
    %     XX(k,i)=signal_after_IFFT(k,i+IFFT_bin_length-GI);%加循环前缀CP,前缀是符号后面的部分
    % end
    XX(k,1:GI)=signal_after_IFFT(k,(IFFT_bin_length-GI+1):IFFT_bin_length);
    % for j=1:GIP
    %     XX(k,IFFT_bin_length+GI+j)=signal_after_IFFT(k,j);%加循环后缀，后缀是符号前面的部分
    % end
    XX(k,(IFFT_bin_length+GI+1):(IFFT_bin_length+GI+GIP))=signal_after_IFFT(k,1:GIP);
end
 
time_wave_matrix_cp=XX;  %iFFT_bin_length+GI+GIP=660
subplot(3,1,2);
plot(0:length(time_wave_matrix_cp)-1,time_wave_matrix_cp(2,:));
axis([0, 700, -0.2, 0.2]);
grid on;  
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal with CP, One Symbol Period');
 
 
 
%% ***************OFDM符号加窗操作******************
windowed_time_wave_matrix_cp=zeros(1,IFFT_bin_length+GI+GIP);
for i = 1:symbols_per_carrier %12
windowed_time_wave_matrix_cp(i,:) = real(time_wave_matrix_cp(i,:)).*rcoswindow(beta,IFFT_bin_length+GI)';%升余弦滚降系数=循环后缀比率
end  
subplot(3,1,3);
plot(0:IFFT_bin_length-1+GI+GIP,windowed_time_wave_matrix_cp(2,:));
axis([0, 700, -0.2, 0.2]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal Apply a Window , One Symbol Period');
  
% 加窗的ofdm符号传输时当前符号的后缀与下一个符号的前缀重合
windowed_Tx_data=zeros(1,symbols_per_carrier*(IFFT_bin_length+GI)+GIP);
windowed_Tx_data(1:IFFT_bin_length+GI+GIP)=windowed_time_wave_matrix_cp(1,:);
for i = 1:symbols_per_carrier-1 
    windowed_Tx_data((IFFT_bin_length+GI)*i+1:(IFFT_bin_length+GI)*(i+1)+GIP)=windowed_time_wave_matrix_cp(i+1,:);%加窗，且后缀与前缀重叠
end
 
%=======================================================
Tx_data_withoutwindow =reshape(time_wave_matrix_cp',(symbols_per_carrier)*(IFFT_bin_length+GI+GIP),1)';%未加窗
Tx_data =reshape(windowed_time_wave_matrix_cp',(symbols_per_carrier)*(IFFT_bin_length+GI+GIP),1)';%加窗，但后缀不与前缀重叠
%=================================================================
%加窗但是后缀不与前缀重叠
temp_time1 = (symbols_per_carrier)*(IFFT_bin_length+GI+GIP);
figure (5)
subplot(2,1,1);
plot(0:temp_time1-1,Tx_data );
grid on
ylabel('Amplitude (volts)')
xlabel('Time (samples)')
title('OFDM Tx_data')
%加窗后缀与前缀重叠
temp_time2 =symbols_per_carrier*(IFFT_bin_length+GI)+GIP;
subplot(2,1,2);
plot(0:temp_time2-1,windowed_Tx_data);
grid on
ylabel('Amplitude (volts)')
xlabel('Time (samples)')
title('OFDM windowed_Tx_data')

%% PAPR 、Power Complementary Cumulative Distribution Function（功率互补累计积分函数） Derivation for OFDM
PAPR = zeros(1,IFFT_bin_length+GI+GIP);
PAPR_windowed = zeros(1,IFFT_bin_length+GI+GIP);
for i = 1:IFFT_bin_length+GI+GIP   %deriving the papr of all subcarriers
    PAPR(i)= 10*log10(max(abs(time_wave_matrix_cp(:,i)).^2)/mean(abs(time_wave_matrix_cp(:,i))).^2);
    PAPR_windowed(i)= 10*log10(max(abs(windowed_time_wave_matrix_cp(:,i)).^2)/mean(abs(windowed_time_wave_matrix_cp(:,i))).^2);
end
[Y1,X1] = hist(PAPR,200);%[区间内元素个数，区间中心数值]=hist[待统计的数据，将数据划分的区间个数]
[Y2,X2] = hist(PAPR_windowed,200);
i = 1:IFFT_bin_length+GI+GIP;
%未加窗
figure(11)
subplot(2,1,1);
plot(i,PAPR,'-b', 'LineWidth',1.2);
title('PAPR of each subcarrier','FontSize',15)
xlabel('subcarrier','FontSize',15)
ylabel('PAPR, dB','FontSize',15)
subplot(2,1,2);
plot(X1,1-cumsum(Y1)/max(cumsum(Y1)),'-b', 'LineWidth',1.2);%cumsum()求累计和1，2，3->1,3,6
title('Power CCDF of OFDM','FontSize',15)
xlabel('PAPR, dB','FontSize',15)
ylabel('Probability','FontSize',15)
%加窗，但后缀不与前缀重叠
figure(12)
subplot(2,1,1);
plot(i,PAPR_windowed,'-b', 'LineWidth',1.2);
title('PAPR of each subcarrier','FontSize',15)
xlabel('subcarrier','FontSize',15)
ylabel('PAPR, dB','FontSize',15)
subplot(2,1,2);
plot(X2,1-cumsum(Y2)/max(cumsum(Y2)),'-b', 'LineWidth',1.2);
title('Power CCDF of OFDM after windowed','FontSize',15)
xlabel('PAPR, dB','FontSize',15)
ylabel('Probability','FontSize',15)

%% =================未加窗作图==================================
symbols_per_average = ceil(symbols_per_carrier/5);% 为什么/5？
avg_temp_time = (IFFT_bin_length+GI+GIP)*symbols_per_average;%为什么取3个ofdm符号做fft？
averages = floor(temp_time1/avg_temp_time);
average_fft(1:avg_temp_time) = 0;
for a = 0:(averages-1)
    subset_ofdm = Tx_data_withoutwindow (((a*avg_temp_time)+1):((a+1)*avg_temp_time));
    subset_ofdm_f = abs(fft(subset_ofdm));
    average_fft = average_fft + (subset_ofdm_f/averages);%频域/整体包含的时间小组？
end
average_fft_log = 20*log10(average_fft);
figure (6)
subplot(2,1,1);
plot((0:(avg_temp_time-1))/avg_temp_time, average_fft_log)% 0/avg_temp_time  :  (avg_temp_time-1)/avg_temp_time
hold on
plot(0:1/IFFT_bin_length:1, -35, 'rd')%-35又是从哪来的常量？
grid on
axis([0 0.5 -40 max(average_fft_log)])
ylabel('Magnitude (dB)')
xlabel('Normalized Frequency (0.5 = fs/2)')
title('OFDM Signal Spectrum without windowing')
%% ===============加窗作图=================================
symbols_per_average = ceil(symbols_per_carrier/5);
avg_temp_time = (IFFT_bin_length+GI+GIP)*symbols_per_average;
averages = floor(temp_time1/avg_temp_time);
average_fft(1:avg_temp_time) = 0;
for a = 0:(averages-1)
    subset_ofdm = Tx_data(((a*avg_temp_time)+1):((a+1)*avg_temp_time));
    subset_ofdm_f = abs(fft(subset_ofdm));
    average_fft = average_fft + (subset_ofdm_f/averages);
end
average_fft_log = 20*log10(average_fft);
subplot(2,1,2)
plot((0:(avg_temp_time-1))/avg_temp_time, average_fft_log)%  0/avg_temp_time  :  (avg_temp_time-1)/avg_temp_time
hold on
plot(0:1/IFFT_bin_length:1, -35, 'rd')
grid on
axis([0 0.5 -40 max(average_fft_log)])
ylabel('Magnitude (dB)')
xlabel('Normalized Frequency (0.5 = fs/2)')
title('Windowed OFDM Signal Spectrum')
%% 上变频



%% ====================添加噪声============================================
Tx_signal_power = var(windowed_Tx_data);%得到发射数据功率
linear_SNR=10^(SNR/10);%信噪比转换成int型
noise_sigma=Tx_signal_power/linear_SNR;%噪声功率
noise_scale_factor = sqrt(noise_sigma);%噪声标准差
noise=randn(1,((symbols_per_carrier)*(IFFT_bin_length+GI))+GIP)*noise_scale_factor;%随机噪声，randn均值为0，方差σ^2 = 1，标准差σ = 1的正态分布的随机数，D（cX）=c^2 * D（X）
 
%noise=wgn(1,length(windowed_Tx_data),noise_sigma,'complex');
 
Rx_data=windowed_Tx_data+noise;%接收数据
%% 下变频



%% 不考虑多径，完美同步，AWGN
%% =====================循环前后缀去除==========================================
Rx_data_matrix=zeros(symbols_per_carrier,IFFT_bin_length+GI+GIP);
for i=1:symbols_per_carrier
    Rx_data_matrix(i,:)=Rx_data(1,(i-1)*(IFFT_bin_length+GI)+1:i*(IFFT_bin_length+GI)+GIP);%将接收信号分为12个符号
end
Rx_data_complex_matrix=Rx_data_matrix(:,GI+1:IFFT_bin_length+GI);%去除CP和CPI


%% =================FFT=================================
Y1=fft(Rx_data_complex_matrix,IFFT_bin_length,2);
% 频域信息
Rx_carriers=Y1(:,carriers);%提取数部分
Rx_phase =angle(Rx_carriers);%获得相位信息
Rx_mag = abs(Rx_carriers);%获得幅度信息
figure(7);
polar(Rx_phase, Rx_mag,'bd');%接收数据的相位和幅度信息作图
title('Phase and mapulitude of Rx_data');
%======================================================================
% 绘图方式1
[M, N]=pol2cart(Rx_phase, Rx_mag); %极坐标转笛卡尔坐标
Rx_complex_carrier_matrix = complex(M, N);%得到复信号
figure(8);
plot(Rx_complex_carrier_matrix,'*r');%接收信号的星坐图
title('star map of Rx_data');
axis([-4, 4, -4, 4]);
grid on
% 绘图方式2
figure(9);
plot(Rx_carriers,'*r');%接收信号的星坐图
title('star map of Rx_data');
axis([-4, 4, -4, 4]);
grid on

%% ********模拟时延、速度信息***********************************************************
% 本质是一个基带传输，并没有上下变频
%*******************距离和速度参数设置*******************
% 单个点目标
V = -400; % m/s
R = 29.25; % m
theta = pi/6;% rad
RX_num = 8;%RX天线通道数
Angle_fft_num = RX_num;%天线维fft点数
lambda = c/f_c;   %ofdm信号波长
d = lambda/2;%天线阵元间距
%*******************构建Kr 和 Kd 向量(参照strum的论文)*******************
kr = zeros(1,carrier_count);
for k = 1:carrier_count
    kr(k) = exp(-1i * 2 * pi * (k-1) * delta_f * 2 * R / c);
end

kd = zeros(1,symbols_per_carrier);
T_OFDM = 1/delta_f * (1 + PrefixRatio);
for k = 1:symbols_per_carrier
    kd(k) = exp(1i * 2 * pi * T_OFDM * (k-1) * 2 * V * f_c / c);
end

ka = zeros(1,RX_num);
for k=1:RX_num
    ka(k) = exp(1i * 2 * pi * (d * sin(theta)/lambda) * (k-1));
end

% Rx_complex_carrier_matrix_radar = Rx_data_complex_matrix .* (kd' *  kr);
% 频域叠加时延多普勒信息，Rx_complex_carrier_matrix是频域形式
gg=kd' *  kr;
Rx_complex_carrier_matrix_radar = 1 * Rx_complex_carrier_matrix .* (kd' *  kr);%时延多普勒项不受fft操作影响，直接乘在频域也行
% 创建多天线接收矩阵
multi_Rx_complex_carrier_matrix_radar = repmat(zeros(size(Rx_complex_carrier_matrix_radar)),[1 1 RX_num]);
for page=1:RX_num
    multi_Rx_complex_carrier_matrix_radar(:,:,page) = Rx_complex_carrier_matrix_radar * ka(page);
end
%% ********OFDM通信信号处理***********************************************************
% 本代码多天线接收但是只进行了单天线解码
%====================4qam解码==================================================
Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
Rx_decoded_binary_symbols=demoduqam4(Rx_serial_complex_symbols);

% %====================8qam解码==================================================
% Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
% Rx_decoded_binary_symbols=demoduqam8(Rx_serial_complex_symbols);

% %====================16qam解码==================================================
% Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
% Rx_decoded_binary_symbols=demoduqam16(Rx_serial_complex_symbols);

% %====================32qam解码==================================================
% Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
% Rx_decoded_binary_symbols=demoduqam32(Rx_serial_complex_symbols);

% %====================64qam解码==================================================
% Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
% Rx_decoded_binary_symbols=demoduqam64(Rx_serial_complex_symbols);


%============================================================
baseband_in = Rx_decoded_binary_symbols;
 
figure(9);
subplot(2,1,1);
stem(baseband_out(1:100));
subplot(2,1,2);
stem(baseband_in(1:100));
title('sending beta， 1-200');
%================计算误比特率=============================================
bit_errors=find(baseband_in ~=baseband_out);
bit_error_count = size(bit_errors, 2);
ber=bit_error_count/baseband_out_length;

%% ********OFDM雷达信号处理***********************************************************
%测距
Range_ifft=zeros(size(multi_Rx_complex_carrier_matrix_radar));
% div = Rx_complex_carrier_matrix_radar(1,:) ./ complex_carrier_matrix(1,:);
% % div = Y1(1,:) ./ IFFT_modulation(1,:);
% div_IFFT = ifft(div);
% Range_fft(1,:)=div_IFFT;
% [max_,index] = max(div_IFFT);
for j=1:RX_num
    div_page = multi_Rx_complex_carrier_matrix_radar (:,:,j) ./ complex_carrier_matrix;
    for i=1:symbols_per_carrier
        div_fft_step = div_page(i,:);
        r_ifft = ifft(div_fft_step);
        Range_ifft(i,:,j)=r_ifft;
    end
end

% 测速
Velocity_fft=zeros(size(multi_Rx_complex_carrier_matrix_radar));
for j=1:RX_num
    for i=1:carrier_count
        r_ifft_step = Range_ifft(:,i,j);
        v_fft_step = fftshift(fft(r_ifft_step));
        Velocity_fft(:,i,j)=v_fft_step;
    end
end

% 测角
Angle_fft=zeros(symbols_per_carrier, carrier_count, Angle_fft_num);
for j=1:symbols_per_carrier
    for i=1:carrier_count
        v_fft_step = Velocity_fft(j, i, :);
        a_fft_step = fftshift(fft(v_fft_step, Angle_fft_num));
        Angle_fft(j, i, :)=a_fft_step;
    end
end

% 计算峰值位置,并估计目标运动状态
Angle_profile=abs(Angle_fft);
peak=max(Angle_profile(:));%单目标取最大值
[index_v,index_r,index_a]=ind2sub(size(Angle_profile),find(Angle_profile==peak));

M_R = ((index_r-1) / carrier_count) * (c / 2 / delta_f)
N_V = ((index_v-symbols_per_carrier/2-1) / symbols_per_carrier) * (c / 2 / f_c/T_OFDM)
T_A = asin((index_a-Angle_fft_num/2-1) / Angle_fft_num*(lambda/d))*(180/pi)

%速度-距离立体图
b=-symbols_per_carrier/2:1:symbols_per_carrier/2-1;
a=1:1:carrier_count;
figure
[A,B] = meshgrid(a.*(c / 2 / delta_f)/carrier_count,b.*(c / 2 / f_c/T_OFDM)/symbols_per_carrier);
mesh(A,B,abs(Velocity_fft(:,:,1)));
xlabel('距离/m');ylabel('速度（m/s）');zlabel('信号幅值');
title('2维FFT处理三维视图');

%距离方位立体图
b=1:1:carrier_count;
a=-Angle_fft_num/2:1:Angle_fft_num/2-1;
figure;
Angle_profile_temp = reshape(sum(Angle_profile, 1),carrier_count,Angle_fft_num);
[X,Y]=meshgrid(asin(a .* (lambda/d) / Angle_fft_num)*180/pi, b.*(c / 2 / delta_f)/carrier_count);
mesh(X,Y,(abs(Angle_profile_temp))); 
xlabel('theta/°');ylabel('距离(m)');zlabel('信号幅值');
title('2维FFT处理三维视图');

% %测距
% figure(10)
% div_IFFT = div_IFFT / symbols_per_carrier;
% plot(abs(div_IFFT));
% text(floor(index),abs(div_IFFT(round(index))),'o','color','r');  %标记出峰值点
% text(index + 5,abs(div_IFFT(round(index))) + 0.02,['(',num2str(index),',',num2str(abs(div_IFFT(round(index)))),')'],'color','r');  %标记出峰值点
% xlabel('index');
% ylabel('Mag');
% title('OFDM radar ranging');
