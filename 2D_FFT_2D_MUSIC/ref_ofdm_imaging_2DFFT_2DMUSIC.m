%% 本代码采用一个ofdm子帧进行点云成像
clc;
clear;
close all;

ref_space = 4;%每4个时隙插入一组导频
slot_num=16;%时隙数
IFFT_length=2048;%子载波资源数
carrier_count=200;%子载波数%%%%%%
ref_carrier_count = IFFT_length/4;%导频频域占用,comb4
comb_num = 4;
symbols_per_carrier=224;%OFDM符号数,在子载波带宽为240kHz的条件下，每个子帧含有16个时隙，16*14个ofdm符号，使用一个子帧
ref_symbol_count = (slot_num/ref_space) * 12;%导频时域占用,每4个时隙插入一组PRS（12个符号）

bits_per_symbol=2;%每个调制符号的比特数（4QAM对应的就是2）
% bits_per_symbol=3;%每个调制符号的比特数（8QAM对应的就是3）
% bits_per_symbol=4;%每个调制符号的比特数（16QAM对应的就是4）
% bits_per_symbol=5;%每个调制符号的比特数（32QAM对应的就是5）
% bits_per_symbol=6;%每个调制符号的比特数（64QAM对应的就是6）


PrefixRatio=1/4;%循环前缀比率 1/6~1/4
GI=PrefixRatio*IFFT_length ;%循环前缀的长度
beta=1/32;%循环后缀比率
GIP=beta*(IFFT_length+GI);%循环后缀长度
SNR=20; %信噪比，单位是dB

%% 基本参数设置
c = 3*10^8;  % 电磁波传播速度， 单位m/s
delta_f = 240*10^3;  % 载波间隔，单位hz
f_c = 70*10^9; % 信号中心频偏, 单位hz, 25GHz,n258频段
%% ================发送数据生成===================================
disp('正在构建发送数据通信信息与PRS索引......');
baseband_out_length = IFFT_length * symbols_per_carrier * bits_per_symbol;%发送比特长度
PSF_num = ref_carrier_count * ref_symbol_count * bits_per_symbol;% 参考信号比特长度
Info_num = baseband_out_length - PSF_num;%通信信号比特长度
% 参考信号RE频域索引
carriers_f = zeros(comb_num, ref_carrier_count);
Info_carriers_f_1 = zeros(comb_num, IFFT_length - ref_carrier_count);%第一类通信信息频域索引
f_offset = [0,2,1,3];
for i=1:comb_num
    offset = f_offset(i);
    for j=1:ref_carrier_count
        carriers_f(i,j) = comb_num*(j-1)+1 + offset;
    end
    Info_carriers_f_1(i, :) = setdiff((1:IFFT_length), carriers_f(i,:));
end
Info_carriers_f_1_full = repmat(Info_carriers_f_1, (12/comb_num) * ref_space, 1);%第一类通信信息频域索引
carriers_f_full = repmat(carriers_f, (12/comb_num) * ref_space, 1);
% 参考信号RE时域索引
symbols_t = zeros(1,ref_symbol_count);
for i=1:slot_num/ref_space
    symbols_t(1,(i-1)*12+1:i*12) = 14*ref_space*(i-1)+2 : 14*ref_space*(i-1)+13;
end
Info_symbol_t_1 = symbols_t;%第一类通信信息时域索引

Info_carriers_f_2 = repmat((1:IFFT_length), (symbols_per_carrier - ref_symbol_count), 1);
Info_symbol_t_2 = setdiff((1:slot_num*14), symbols_t);% 第二类通信资源时域索引

% 参考信号索引
ref_t_f_index = zeros(size(symbols_t,2), size(carriers_f_full,2), 2);
ref_t_f_index(:, :, 1)=repmat(symbols_t', 1, ref_carrier_count);
ref_t_f_index(:, :, 2)=carriers_f_full;

% 通信信号索引
% class 1
Info_t_f_index_1 = zeros(size(Info_symbol_t_1,2), size(Info_carriers_f_1_full,2), 2);
Info_t_f_index_1(:, :, 1)=repmat(Info_symbol_t_1', 1, (IFFT_length - ref_carrier_count));
Info_t_f_index_1(:, :, 2)=Info_carriers_f_1_full;
%class 2
Info_t_f_index_2 = zeros(size(Info_symbol_t_2,2), size(Info_carriers_f_2,2), 2);
Info_t_f_index_2(:, :, 1)=repmat(Info_symbol_t_2', 1, IFFT_length);
Info_t_f_index_2(:, :, 2)=Info_carriers_f_2;
disp('发送数据通信信息与PRS索引构建完毕！');
%% ***************模拟随机通信信息***************
% baseband_out=zeros(1,baseband_out_length);%纯0序列

%% ***************模拟随机通信信息***************
% baseband_out=ones(1,baseband_out_length);%纯1序列

%% ***************模拟随机通信信息***************
% rand( 'twister',0); %随机种子，可以让每次生成的随机数有规律，有了这句话每次生成的随机数具有一样的规律
% baseband_out=round(rand(1,baseband_out_length));%生成随机数（模拟随机的通信数据）
%% ***************m序列模拟的随机通信信息***************
disp('正在使用m序列填充通信信息数据比特......');
Tx_matric = zeros(IFFT_length*bits_per_symbol,symbols_per_carrier);
for i = 1:symbols_per_carrier
    Order_number=12; %m序列的阶数等于10，m序列长度为2^10 - 1
    mg = zeros(IFFT_length*bits_per_symbol,1);
%    生成m序列本源多项式的系数，可以用primpoly(Order_number,'all')得到；
    tmp = primpoly(Order_number,'all','nodisplay'); %生成所有可行的m序列的本源多项式系数，阶数越大，数越多(常数项系数不考虑)
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
    mg(1:IFFT_length*bits_per_symbol-1,1) = tmp(1:IFFT_length*bits_per_symbol-1);
    Tx_matric(:,i) = mg;
end
Tx_matric = Tx_matric';
disp('通信信息数据比特填充完毕！');
%% ****************gold序列模拟PRS信息*****************************
disp('正在使用gold序列填充PRS数据比特......');
    for m=1:ref_symbol_count
        n_slot=ceil(symbols_t(m)/14)-1;
        seq=goldseq(n_slot,symbols_t(m)-1);%伪随机序列
        for k=1:ref_carrier_count
           Tx_matric(ref_t_f_index(m,k,1), 2*ref_t_f_index(m,k,2)-1) = seq(2*ref_t_f_index(m,k,2)-1);
           Tx_matric(ref_t_f_index(m,k,1), 2*ref_t_f_index(m,k,2)) = seq(2*ref_t_f_index(m,k,2));
        end
    end

baseband_out = reshape(Tx_matric',1,baseband_out_length); 
disp('PRS数据比特填充完毕......');
%% *************************数据调制****************************
% ==============4QAM调制====================================
disp('正在进行数据调制，耗时较长,请稍后......');
complex_carrier_matrix=qam4(baseband_out);
complex_carrier_matrix=reshape(complex_carrier_matrix',IFFT_length, symbols_per_carrier)';%symbols_per_carrier*carrier_count
 
figure
plot(complex_carrier_matrix,'*r');
title('star map of Tx_data');
axis([-2, 2, -2, 2]);
grid on
% % ==============8QAM调制====================================
% complex_carrier_matrix=qam8(baseband_out);
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-2, 2, -2, 2]);
% grid on


% % ==============16QAM调制====================================
% complex_carrier_matrix=qam16(baseband_out);
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-4, 4, -4, 4]);
% grid on

% % ==============32QAM调制====================================
% complex_carrier_matrix=qam32(baseband_out);
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-8, 8, -8, 8]);
% grid on

% % ==============64QAM调制====================================
% complex_carrier_matrix=qam64(baseband_out);
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-8, 8, -8, 8]);
% grid on

disp('调制完毕！');
%% =================IFFT===========================
disp('正在使用IFFT生成时域OFDM符号......');
IFFT_modulation=complex_carrier_matrix;
signal_after_IFFT=ifft(IFFT_modulation,IFFT_length,2);%ifft

figure
subplot(3,1,1);
plot(0:IFFT_length-1,signal_after_IFFT(2,:));
axis([0, 3000, -0.2, 0.2]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal, One Symbol Period');
disp('时域OFDM符号生成完毕！');
%% =====================加循环前缀CP和后缀=========================
disp('正在添加循环前缀后缀......');
time_wave_matrix_cp=zeros(symbols_per_carrier,IFFT_length+GI+GIP);%GI=512,GIP=80
for k=1:symbols_per_carrier    %224
    time_wave_matrix_cp(k,GI+1:GI+IFFT_length)=signal_after_IFFT(k,:);
    time_wave_matrix_cp(k,1:GI)=signal_after_IFFT(k,(IFFT_length-GI+1):IFFT_length);%加循环前缀CP,前缀是符号后面的部分
    time_wave_matrix_cp(k,(IFFT_length+GI+1):(IFFT_length+GI+GIP))=signal_after_IFFT(k,1:GIP);%加循环后缀，后缀是符号前面的部分
end
subplot(3,1,2);
plot(0:length(time_wave_matrix_cp)-1,time_wave_matrix_cp(2,:));
axis([0, 3000, -0.2, 0.2]);
grid on;  
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal with CP, One Symbol Period');
disp('循环前缀后缀添加完毕！');
%% ***************OFDM符号加窗操作******************
disp('正在时域加窗并进行并串转换......');
windowed_time_wave_matrix_cp=zeros(symbols_per_carrier,IFFT_length+GI+GIP);
for i = 1:symbols_per_carrier %224
    windowed_time_wave_matrix_cp(i,:) = time_wave_matrix_cp(i,:).*rcoswindow(beta,IFFT_length+GI)';%升余弦滚降系数=循环后缀比率
end  
subplot(3,1,3);
plot(0:IFFT_length-1+GI+GIP,windowed_time_wave_matrix_cp(2,:));
axis([0, 3000, -0.2, 0.2]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal Apply a Window , One Symbol Period');
  
% 并串转换，加窗的ofdm符号传输时当前符号的后缀与下一个符号的前缀重合
windowed_Tx_data=zeros(1,symbols_per_carrier*(IFFT_length+GI)+GIP);
windowed_Tx_data(1:IFFT_length+GI+GIP)=windowed_time_wave_matrix_cp(1,:);
for i = 1:symbols_per_carrier-1 
    windowed_Tx_data((IFFT_length+GI)*i+1:(IFFT_length+GI)*(i+1)+GIP)=windowed_time_wave_matrix_cp(i+1,:);%加窗，且后缀与前缀重叠
end
%加窗后缀与前缀重叠
figure
temp_time2 =symbols_per_carrier*(IFFT_length+GI)+GIP;
plot(0:temp_time2-1,windowed_Tx_data);
grid on
ylabel('Amplitude (volts)')
xlabel('Time (samples)')
title('OFDM windowed_Tx_data')
disp('串行发送信号生成成功！');
%% 上变频

%% ====================添加噪声============================================
disp('正在添加AWGN......');
Tx_signal_power = var(windowed_Tx_data);%得到发射数据功率
linear_SNR=10^(SNR/10);%信噪比转换成int型
noise_sigma=Tx_signal_power/linear_SNR;%噪声功率
noise_scale_factor = sqrt(noise_sigma);%噪声标准差
noise=randn(1,((symbols_per_carrier)*(IFFT_length+GI))+GIP)*noise_scale_factor;%随机噪声，randn均值为0，方差σ^2 = 1，标准差σ = 1的正态分布的随机数，D（cX）=c^2 * D（X）
%noise=wgn(1,length(windowed_Tx_data),noise_sigma,'complex');
Rx_data=windowed_Tx_data+noise;%接收数据
disp('AWGN添加完毕！');
%% 下变频

%% 不考虑多径，完美同步，AWGN
%% =====================串并转换与循环前后缀去除==========================================
disp('正在进行串并转换与循环前后缀消除......');
Rx_data_matrix=zeros(symbols_per_carrier,IFFT_length+GI+GIP);
%串并转换
for i=1:symbols_per_carrier
    Rx_data_matrix(i,:)=Rx_data(1,(i-1)*(IFFT_length+GI)+1:i*(IFFT_length+GI)+GIP);%将接收信号分为224个符号
end
%循环前后缀去除
Rx_data_complex_matrix=Rx_data_matrix(:,GI+1:IFFT_length+GI);%去除CP和CPI
disp('并行信号恢复成功！');
%% =================FFT=================================
disp('正在使用FFT恢复频域信息......');
Y1=fft(Rx_data_complex_matrix,IFFT_length,2);
% 频域信息
Rx_carriers=Y1;%提取数部分
Rx_phase =angle(Rx_carriers);%获得相位信息
Rx_mag = abs(Rx_carriers);%获得幅度信息
figure
polar(Rx_phase, Rx_mag,'bd');%接收数据的相位和幅度信息作图
title('Phase and mapulitude of Rx_data');
%======================================================================
% 绘图方式1
[M, N]=pol2cart(Rx_phase, Rx_mag); %极坐标转笛卡尔坐标
Rx_complex_carrier_matrix = complex(M, N);%得到复信号
figure
plot(Rx_complex_carrier_matrix,'*r');%接收信号的星坐图
title('star map of Rx_data');
axis([-4, 4, -4, 4]);
grid on
% % 绘图方式2
% figure
% plot(Rx_carriers,'*r');%接收信号的星坐图
% title('star map of Rx_data');
% axis([-4, 4, -4, 4]);
% grid on
disp('频域信息恢复成功！');
%% ********OFDM通信信号处理***********************************************************
disp('正在解调获取原始数据比特......');
% 本代码多天线接收但是只进行了单天线解码
%====================4qam解码==================================================
Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix', 1, size(Rx_complex_carrier_matrix,1)*size(Rx_complex_carrier_matrix,2))';
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
disp('解调完毕！');
figure
subplot(2,1,1);
stem(baseband_out(1:100));
subplot(2,1,2);
stem(baseband_in(1:100));
title('sending beta， 1-200');
%================计算误比特率=============================================
bit_errors=find(baseband_in ~=baseband_out);
bit_error_count = size(bit_errors, 2);
disp('误码率为：');
BER=bit_error_count/baseband_out_length
%% ********模拟时延、速度信息***************************************
% 本质是一个基带传输，并没有上下变频
%*******************距离和速度参数设置*******************
disp('正在模拟速度时延与方位信息......');
% 散射环境
environment_point=environment();
% 基站天线位置
base_pos=[14,100,20];
% 环境散射点信息
point_info=zeros(size(environment_point,1),4);
base_pos_full=repmat(base_pos,size(environment_point,1),1);
% 距离
R_info=((environment_point(:,1)-base_pos_full(:,1)).^2+(environment_point(:,2)-base_pos_full(:,2)).^2+(environment_point(:,3)-base_pos_full(:,3)).^2).^(1/2);
% 速度
V_info=environment_point(:,4);
% 角度
xoy_dis=((environment_point(:,1)-base_pos_full(:,1)).^2+(environment_point(:,2)-base_pos_full(:,2)).^2).^(1/2);
A1_info=acos((base_pos_full(:,1)-environment_point(:,1))./xoy_dis);
A2_info=acos((base_pos_full(:,3)-environment_point(:,3))./R_info);
point_info(:,1)=R_info;
point_info(:,2)=V_info;
point_info(:,3)=A1_info;
point_info(:,4)=A2_info;
disp('速度、时延、方位信息模拟完毕！');
%% 回波信号构建（速度距离）
disp('正在生成单天线回波信号，可能耗时较长，请稍后......');
% 构建虚拟接收阵列，采用4发64*64接收，虚拟接收阵列为256*256
M = 16;         %x方向阵源数
N = 16;        %y方向阵源数
lambda = c/f_c;   %ofdm信号波长
K_sub = 8;    %子阵元数目
d = lambda/2;%天线阵元间距
%*******************构建kr、kd、ka 向量*******************
T_OFDM = 1/delta_f * (1 + PrefixRatio);
% 创建多天线接收矩阵
% multi_Rx_complex_carrier_matrix_radar_RX1 = zeros(size(Rx_complex_carrier_matrix));
multi_Rx_complex_carrier_matrix_radar = zeros(symbols_per_carrier, IFFT_length, M, N);

win = waitbar(0, '回波计算中...');
tCount1 = 0;
for tgt_index = 1:size(point_info, 1)
    % 计时初始化
    t00 = tic;
    % 获取单目标信息
    R = point_info(tgt_index, 1); % 目标距离
    V = point_info(tgt_index, 2); % 目标速度
    theta = point_info(tgt_index, 3); % 目标方位角
    faii = point_info(tgt_index,4); % 目标俯仰角

    % 单目标距离信息
    kr = zeros(1,IFFT_length);
    for k = 1:IFFT_length
        kr(k) = exp(-1i * 2 * pi * (k-1) * delta_f * 2 * R / c);
    end
    % 单目标速度信息
    kd = zeros(1,symbols_per_carrier);
    for k = 1:symbols_per_carrier
        kd(k) = exp(1i * 2 * pi * T_OFDM * (k-1) * 2 * V * f_c / c);
    end
    
    % 频域叠加时延多普勒信息，Rx_complex_carrier_matrix是频域形式
    Rx_complex_carrier_matrix_radar = 1 * Rx_complex_carrier_matrix .* (kd' *  kr);%时延多普勒项不受fft操作影响，直接乘在频域也行
    
    %多天线单目标角度信息
    ka=zeros(M,N);
    for index_x=1:M
        for index_y=1:N
              % 波程差为-，且距离关系发生变化
            if theta>(90*pi/180)
                if faii<=(90*pi/180)
                    r=(index_x-1)*d*cos(pi-theta)-(index_y-1)*d*sin(pi-theta);%波程差水平投影
                    ka(index_x,index_y)=exp(-1j*2*pi*r*cos(faii)/lambda); 
                end
                if faii>(90*pi/180)
                    r=(index_x-1)*d*cos(pi-theta)+(index_y-1)*d*sin(pi-theta);%波程差水平投影
                    ka(index_x,index_y)=exp(-1j*2*pi*r*(cos(pi-faii))/lambda); 
                end
            end
              %波程差为+
            if theta<=(90*pi/180)
                
                if faii<=(90*pi/180)
                    r=(index_x-1)*d*cos(theta)+(index_y-1)*d*sin(theta);%波程差水平投影
                    ka(index_x,index_y)=exp(1j*2*pi*r*cos(faii)/lambda);
                end
                if faii>(90*pi/180)
                    r=(index_x-1)*d*cos(theta)-(index_y-1)*d*sin(theta);%波程差水平投影
                    ka(index_x,index_y)=exp(1j*2*pi*r*(cos(pi-faii))/lambda); 
                end
                 
            end
            multi_Rx_complex_carrier_matrix_radar(:,:,index_x,index_y) = multi_Rx_complex_carrier_matrix_radar(:,:,index_x,index_y) + Rx_complex_carrier_matrix_radar * ka(index_x,index_y);
        end 
    end
    % multi_Rx_complex_carrier_matrix_radar_RX1 = multi_Rx_complex_carrier_matrix_radar_RX1 + Rx_complex_carrier_matrix_radar * ka(1,1);
    % 剩余时间预估
    tCount1 = tCount1 + toc(t00);
    t_step = tCount1/tgt_index;
    t_res = (size(point_info, 1) - tgt_index) * t_step;
    str=['剩余运行时间：',num2str(t_res/60),'min'];
    waitbar(tgt_index/size(point_info, 1), win, str)
end
close(win);
disp('单天线回波信号生成完毕！')
%% ********OFDM雷达信号处理（测速测距）***********************************************************
disp('开始测速测距......')

% 测速测距
Velocity_fft = zeros(size(multi_Rx_complex_carrier_matrix_radar));

win = waitbar(0, '正在为所有天线回波进行fft...');
tCount1 = 0;
for i=1:M
    % 计时初始化
    t00 = tic;
    for j=1:N
        div_page = multi_Rx_complex_carrier_matrix_radar(:, :, i, j) ./ complex_carrier_matrix;
        page_ifft = ifft(div_page, IFFT_length, 2);
        page_fft = fftshift(fft(page_ifft, symbols_per_carrier, 1), 1);
        Velocity_fft(:, :, i, j) = page_fft;
    end
    %剩余时间预估
    tCount1 = tCount1 + toc(t00);
    t_step = tCount1/i;
    t_res = (M - i) * t_step;
    str=['剩余运行时间：',num2str(t_res/60),'min'];
    waitbar(i/M, win, str)
end
close(win);

% 首先针对测速测距结果进行恒虚警检测
[RD_threshold_matrix,RD_target_index,RD_detect_matrix_abs] = OSCA_CFAR(Velocity_fft(:, :, 1, 1));
disp('测速测距完毕！')
%% 测速测距结果及CFAR门限绘图
b=-symbols_per_carrier/2:1:symbols_per_carrier/2-1;
a=1:1:IFFT_length;
figure
[A,B] = meshgrid(a.*(c / 2 / delta_f)/IFFT_length,b.*(c / 2 / f_c/T_OFDM)/symbols_per_carrier);
mesh(A,B,RD_detect_matrix_abs);
axis([50 150 -50 50 0 5e6])
xlabel('距离/m');ylabel('速度（m/s）');zlabel('信号幅值');
title('速度距离fft结果');

b_1=-symbols_per_carrier/2:1:symbols_per_carrier/2-1;
a_1=1:1:IFFT_length;
figure
% hold on
[A_1,B_1] = meshgrid(a_1.*(c / 2 / delta_f)/IFFT_length,b_1.*(c / 2 / f_c/T_OFDM)/symbols_per_carrier);
mesh(A_1,B_1,RD_threshold_matrix);
axis([50 150 -50 50 0 5e6])
xlabel('距离/m');ylabel('速度（m/s）');zlabel('信号幅值');
title('速度距离fft门限结果');

%% 回波信号构建（角度）考虑到天线数量较多，因此先通过OSCA_CFAR确定目标存在的RE，再针对对应RE生成多天线接收矩阵，节省内存减小计算压力，同时降低角度分辨率要求
disp('正在构建多天线角度回波矩阵......');
Angel_page_num = size(RD_target_index, 1); % 选取目标对应的RE进行测角
Angle_matrix = zeros(M, N, Angel_page_num);

for i=1:Angel_page_num
    Angle_matrix(:, :, i) = Velocity_fft(RD_target_index(i,1), RD_target_index(i,2), :, :);
end
disp('多天线角度回波信号生成完毕！');
%% 测角
disp('开始测角......')
% MUSIC角度搜索基本参数
space = 0.2; % 搜索粒度
% 搜索范围
theta_head_offset = 60;
theta_back_offset = 60;
faii_head_offset = 60;
faii_back_offset = 90;
theta_list = space + theta_head_offset: space: 180 - theta_back_offset;
faii_list = space + faii_head_offset: space: 180 - faii_back_offset;

Angle_music_matrix = zeros(length(theta_list), length(faii_list), Angel_page_num);
Angle_music_threshold_matrix = zeros(length(theta_list), length(faii_list), Angel_page_num);
Angle_music_abs_matrix = zeros(length(theta_list), length(faii_list), Angel_page_num);
A2_Angle_target_cell = cell(size(RD_target_index, 1), 1);

win = waitbar(0, 'RE目标检测中...');
tCount1 = 0;
for i=1:Angel_page_num
    % 计时初始化
    t00 = tic;
    % 首先取第一行第一列构建协方差矩阵（搜索基准，导向矢量）
    W = Angle_matrix(:, :, i);
    W_azimuth = W(:, 1);
    W_pitch = W(1, :).';
    % 未经平滑
    R_azimuth_ns = W_azimuth * W_azimuth';
    R_pitch_ns = W_pitch * W_pitch';
    % 空间平滑算法
    R_azimuth = smooth_covariance(R_azimuth_ns, K_sub);
    R_pitch = smooth_covariance(R_pitch_ns, K_sub);
    
    
    [EV_azimuth,D_azimuth] = eig(R_azimuth); %拿到特向量EV + 特征值D  新版本matlab已经从小到大排序好了
    diag_azimuth = diag(D_azimuth);
    signal_space_num_azimuth = WCA_CFAR_1D(diag_azimuth);
    En_azimuth = EV_azimuth(:, 1:(K_sub-signal_space_num_azimuth)); %signal_space_num_azimuth这里代表的是信号子空间维度，这里直接给了，具体的确定方法：1陈旭师兄的特征值相除比值比较2我的恒虚警检测
    % figure
    % bar3(D_azimuth);
    % title('azimuth特征值矩阵排列')
    
    [EV_pitch,D_pitch] = eig(R_pitch); %拿到特向量EV + 特征值D  新版本matlab已经从小到大排序好了
    diag_pitch = diag(D_pitch);
    signal_space_num_pitch = WCA_CFAR_1D(diag_pitch);
    En_pitch = EV_pitch(:, 1:(K_sub-signal_space_num_pitch)); %signal_space_num_pitch这里代表的是信号子空间维度，这里直接给了，具体的确定方法：1陈旭师兄的特征值相除比值比较2我的恒虚警检测
    % figure
    % bar3(D_pitch);
    % title('pitch特征值矩阵排列')

    for theta_index=1:length(theta_list)
    theta_search = theta_list(theta_index)*pi/180;
        for faii_index=1:length(faii_list)
            faii_search = faii_list(faii_index)*pi/180;
            W_search = zeros(K_sub,K_sub); % 阵列流形归零
    
            for index_x=1:K_sub
                for index_y=1:K_sub
                    % 波程差为-，且距离关系发生变化
                    if theta_search>(90*pi/180)
                        if faii_search<=(90*pi/180)
                            r=(index_x-1)*d*cos(pi-theta_search)-(index_y-1)*d*sin(pi-theta_search);%波程差水平投影
                            W_search(index_x,index_y)=exp(-1j*2*pi*r*cos(faii_search)/lambda); 
                        end
                        if faii_search>(90*pi/180)
                            r=(index_x-1)*d*cos(pi-theta_search)+(index_y-1)*d*sin(pi-theta_search);%波程差水平投影
                            W_search(index_x,index_y)=exp(-1j*2*pi*r*(cos(pi-faii_search))/lambda); 
                        end
                    end
                    %波程差为+
                    if theta_search<=(90*pi/180)
    
                        if faii_search<=(90*pi/180)
                            r=(index_x-1)*d*cos(theta_search)+(index_y-1)*d*sin(theta_search);%波程差水平投影
                            W_search(index_x,index_y)=exp(1j*2*pi*r*cos(faii_search)/lambda);
                        end
                        if faii_search>(90*pi/180)
                            r=(index_x-1)*d*cos(theta_search)-(index_y-1)*d*sin(theta_search);%波程差水平投影
                            W_search(index_x,index_y)=exp(1j*2*pi*r*(cos(pi-faii_search))/lambda); 
                        end
                    end
                end 
            end
    
            W_search_azimuth = W_search(:,1);
            W_search_pitch = W_search(1, :).';
    
            o_matrix_azimuth = (W_search_azimuth'*En_azimuth)*(En_azimuth'*W_search_azimuth);
            o_matrix_pitch = (W_search_pitch'*En_pitch)*(En_pitch'*W_search_pitch);
            
            o_matrix_azimuth_m(theta_index, faii_index) = abs(1./o_matrix_azimuth);
            o_matrix_pitch_m(theta_index, faii_index) = abs(1./o_matrix_pitch);
            % music_matrix(theta_index+1, faii_index+1) = abs(1./o_matrix_azimuth) * abs(1./o_matrix_pitch);
        end
    end
    Angle_music_matrix(:, :, i)= o_matrix_azimuth_m .* o_matrix_pitch_m;
    % 两次music匹配情况
    % figure
    % mesh(o_matrix_azimuth_m)
    % figure
    % mesh(o_matrix_pitch_m)
    % % 整体结果
    % figure
    % a=1:size(Angle_music_matrix(:, :, i), 2);
    % b=1:size(Angle_music_matrix(:, :, i), 1);
    % [X,Y]=meshgrid(a,b);
    % mesh(X*space+faii_head_offset,Y*space+theta_head_offset,music_matrix);
    % xlabel('俯仰角faii/°')
    % ylabel('方位角theta/°')

    % CA_CFAR检测
    [A2_threshold_matrix,A2_target_index,A2_detect_matrix_abs] = CA_CFAR(Angle_music_matrix(:, :, i));
    Angle_music_threshold_matrix(:, :, i) = A2_threshold_matrix;
    Angle_music_abs_matrix(:, :, i) = A2_detect_matrix_abs;
    A2_Angle_target_cell{i, 1} = A2_target_index;
    
    X = ['第', num2str(i), '个目标RE检测完毕'];
    disp(X);
    %剩余时间预估
    tCount1 = tCount1 + toc(t00);
    t_step = tCount1/i;
    t_res = (Angel_page_num - i) * t_step;
    str=['剩余运行时间：',num2str(t_res/60),'min'];
    waitbar(i/Angel_page_num, win, str)
end
close(win);
disp('测角完毕！');
%% 测角结果及CFAR门限绘图
check_num = 6;
figure
a=1:size(Angle_music_abs_matrix(:, :, check_num), 2);
b=1:size(Angle_music_abs_matrix(:, :, check_num), 1);
[X,Y]=meshgrid(a,b);
mesh(X*space+faii_head_offset,Y*space+theta_head_offset,abs(Angle_music_abs_matrix(:, :, check_num)));
axis([60 90 60 120])
title('方位music结果');
xlabel('俯仰角faii/°')
ylabel('方位角theta/°')

figure
a=1:size(Angle_music_threshold_matrix(:, :, check_num), 2);
b=1:size(Angle_music_threshold_matrix(:, :, check_num), 1);
[X,Y]=meshgrid(a,b);
mesh(X*space+faii_head_offset,Y*space+theta_head_offset,abs(Angle_music_threshold_matrix(:, :, check_num)));
axis([60 90 60 120])
title('方位music门限');
xlabel('俯仰角faii/°')
ylabel('方位角theta/°')
%% 恢复原始位置信息
pos_all = [];
angle_all= [];
for i=1:size(RD_target_index,1)
    % 速度距离计算
    M_R = ((RD_target_index(i, 2)-1) / IFFT_length) * (c / 2 / delta_f);
    N_V = -((RD_target_index(i, 1)-symbols_per_carrier/2-1) / symbols_per_carrier) * (c / 2 / f_c / T_OFDM);
    % 空间角度查询
    for j=1:size(A2_Angle_target_cell{i, 1}, 1)
        theta_estimation = A2_Angle_target_cell{i, 1}(j,1)*space + theta_head_offset;
        faii_estimation = A2_Angle_target_cell{i, 1}(j,2)*space + faii_head_offset;

        angle_all=[angle_all; theta_estimation faii_estimation];

        % 恢复笛卡尔坐标系信息
        pos_z = base_pos(3) - M_R * cosd(faii_estimation);
        pos_x = base_pos(1) + M_R * sind(faii_estimation) * cosd(theta_estimation);
        pos_y = base_pos(2) - M_R * sind(faii_estimation) * sind(theta_estimation);
        % [pos_x, pos_y] = pol2cart(faii_estimation, M_R * sin(theta_estimation));
        % 储存位置信息
        pos_all = [pos_all;pos_x pos_y pos_z N_V];
    end
end

figure
x=pos_all(:,1);
y=pos_all(:,2);
z=pos_all(:,3);
c=pos_all(:,4);

scatter3(x,y,z,50,c,'.');
axis([0 30 0 20 0 20])
xlabel('X/m')
ylabel('Y/m')
grid on
h = colorbar;%右侧颜色栏
set(get(h,'label'),'string','运动速度');%给右侧颜色栏命名

%% 相同绘图方式验证绘图流程、散射点信息创建流程（没问题）
pos_all_true = [];
for i=1:size(A1_info, 1)
    theta_true = point_info(i, 3);
    faii_true = point_info(i, 4);
    R_true = point_info(i, 1);
    v_true = point_info(i, 2);

    % 恢复笛卡尔坐标系信息
    pos_z_true = base_pos(3) - R_true * cos(faii_true);
    pos_x_true = base_pos(1) + R_true * sin(faii_true) * cos(theta_true);
    pos_y_true = base_pos(2) - R_true * sin(faii_true) * sin(theta_true);
    % [pos_x, pos_y] = pol2cart(faii_estimation, M_R * sin(theta_estimation));
    % 储存位置信息
    pos_all_true = [pos_all_true;pos_x_true pos_y_true pos_z_true v_true];
end
figure
x=pos_all_true(:,1);
y=pos_all_true(:,2);
z=pos_all_true(:,3);
c=pos_all_true(:,4);

scatter3(x,y,z,50,c,'.');
axis([0 30 0 20 0 20])
xlabel('X/m')
ylabel('Y/m')
grid on
h = colorbar;%右侧颜色栏
set(get(h,'label'),'string','运动速度');%给右侧颜色栏命名