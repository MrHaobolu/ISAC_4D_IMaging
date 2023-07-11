%% 
clear;
close all;
carrier_count=200;%���ز���
symbols_per_carrier=12;%OFDM������

bits_per_symbol=2;%ÿ�����Ʒ��ŵı�������4QAM��Ӧ�ľ���2��
% bits_per_symbol=3;%ÿ�����Ʒ��ŵı�������8QAM��Ӧ�ľ���3��
% bits_per_symbol=4;%ÿ�����Ʒ��ŵı�������16QAM��Ӧ�ľ���4��
% bits_per_symbol=5;%ÿ�����Ʒ��ŵı�������32QAM��Ӧ�ľ���5��
% bits_per_symbol=6;%ÿ�����Ʒ��ŵı�������64QAM��Ӧ�ľ���6��

IFFT_bin_length=512;%FFT����
PrefixRatio=1/4;%ѭ��ǰ׺���� 1/6~1/4
GI=PrefixRatio*IFFT_bin_length ;%ѭ��ǰ׺�ĳ���
beta=1/32;%ѭ����׺����
GIP=beta*(IFFT_bin_length+GI);%ѭ����׺����
SNR=20; %����ȣ���λ��dB

%% ������������

c = 3*10^8;  % ��Ų������ٶȣ� ��λm/s
delta_f = 240*10^3;  % �ز��������λhz
f_c = 24*10^9; % �ź�����Ƶƫ, ��λhz

%% ================������������===================================
baseband_out_length = carrier_count * symbols_per_carrier * bits_per_symbol;%���ͱ��س���  200*12*4
carriers = (1:carrier_count) + (floor(IFFT_bin_length/4) - floor(carrier_count/2));%-floor(carrier_count/2)�����ز��е��Ƶ�0�㣬+floor(IFFT_bin_length/4)�����ز����ݰᵽ512ǰ�벿�ֵ�����
conjugate_carriers = IFFT_bin_length - carriers + 2;
%% ***************ģ�����ͨ����Ϣ***************
% baseband_out=zeros(1,baseband_out_length);%��0����

%% ***************ģ�����ͨ����Ϣ***************
% baseband_out=ones(1,baseband_out_length);%��1����

%% ***************ģ�����ͨ����Ϣ***************
rand( 'twister',0); %������ӣ�������ÿ�����ɵ�������й��ɣ�������仰ÿ�����ɵ����������һ���Ĺ���
baseband_out=round(rand(1,baseband_out_length));%�����������ģ�������ͨ�����ݣ�
%% ***************m����***************
%==================================================
Tx_matric = zeros(carrier_count*bits_per_symbol,symbols_per_carrier);
for i = 1:symbols_per_carrier
    Order_number=10; %m���еĽ�������10��m���г���Ϊ2^10 - 1
    mg = zeros(carrier_count*bits_per_symbol,1);
%    mg(1:carrier_count*bits_per_symbol-1,1)  =idinput((2^(Order_number)-1),'prbs')';%����idinput�����������ɶ�������
%    ����m���б�Դ����ʽ��ϵ����������primpoly(Order_number,'all')�õ���
    tmp = primpoly(Order_number,'all'); %�������п��е�m���еı�Դ����ʽϵ��������Խ����Խ��(������ϵ��������)
    cur_tmp = int32(tmp(1)); %ѡ���һ��
    % ʮ���ƻ�Ϊ������
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


%% ==============4QAM����====================================

complex_carrier_matrix=qam4(baseband_out);
 
complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
 
figure(1);
plot(complex_carrier_matrix,'*r');
title('star map of Tx_data');
axis([-2, 2, -2, 2]);
grid on

% %% ==============8QAM����====================================
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


% %% ==============16QAM����====================================
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

% %% ==============32QAM����====================================
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

% %% ==============64QAM����====================================
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
%ͨ����������Ϣ��ֵ��ofdm������Ӧ���ز���
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
 
%% =====================��ѭ��ǰ׺CP�ͺ�׺====================================
XX=zeros(symbols_per_carrier,IFFT_bin_length+GI+GIP);%GI=128,GIP=20
for k=1:symbols_per_carrier    %12
    % for i=1:IFFT_bin_length         %512
    %     XX(k,i+GI)=signal_after_IFFT(k,i);%129--640
    % end
    XX(k,GI+1:GI+IFFT_bin_length)=signal_after_IFFT(k,:);
    % for i=1:GI %1--128
    %     XX(k,i)=signal_after_IFFT(k,i+IFFT_bin_length-GI);%��ѭ��ǰ׺CP,ǰ׺�Ƿ��ź���Ĳ���
    % end
    XX(k,1:GI)=signal_after_IFFT(k,(IFFT_bin_length-GI+1):IFFT_bin_length);
    % for j=1:GIP
    %     XX(k,IFFT_bin_length+GI+j)=signal_after_IFFT(k,j);%��ѭ����׺����׺�Ƿ���ǰ��Ĳ���
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
 
 
 
%% ***************OFDM���żӴ�����******************
windowed_time_wave_matrix_cp=zeros(1,IFFT_bin_length+GI+GIP);
for i = 1:symbols_per_carrier %12
windowed_time_wave_matrix_cp(i,:) = real(time_wave_matrix_cp(i,:)).*rcoswindow(beta,IFFT_bin_length+GI)';%�����ҹ���ϵ��=ѭ����׺����
end  
subplot(3,1,3);
plot(0:IFFT_bin_length-1+GI+GIP,windowed_time_wave_matrix_cp(2,:));
axis([0, 700, -0.2, 0.2]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal Apply a Window , One Symbol Period');
  
% �Ӵ���ofdm���Ŵ���ʱ��ǰ���ŵĺ�׺����һ�����ŵ�ǰ׺�غ�
windowed_Tx_data=zeros(1,symbols_per_carrier*(IFFT_bin_length+GI)+GIP);
windowed_Tx_data(1:IFFT_bin_length+GI+GIP)=windowed_time_wave_matrix_cp(1,:);
for i = 1:symbols_per_carrier-1 
    windowed_Tx_data((IFFT_bin_length+GI)*i+1:(IFFT_bin_length+GI)*(i+1)+GIP)=windowed_time_wave_matrix_cp(i+1,:);%�Ӵ����Һ�׺��ǰ׺�ص�
end
 
%=======================================================
Tx_data_withoutwindow =reshape(time_wave_matrix_cp',(symbols_per_carrier)*(IFFT_bin_length+GI+GIP),1)';%δ�Ӵ�
Tx_data =reshape(windowed_time_wave_matrix_cp',(symbols_per_carrier)*(IFFT_bin_length+GI+GIP),1)';%�Ӵ�������׺����ǰ׺�ص�
%=================================================================
%�Ӵ����Ǻ�׺����ǰ׺�ص�
temp_time1 = (symbols_per_carrier)*(IFFT_bin_length+GI+GIP);
figure (5)
subplot(2,1,1);
plot(0:temp_time1-1,Tx_data );
grid on
ylabel('Amplitude (volts)')
xlabel('Time (samples)')
title('OFDM Tx_data')
%�Ӵ���׺��ǰ׺�ص�
temp_time2 =symbols_per_carrier*(IFFT_bin_length+GI)+GIP;
subplot(2,1,2);
plot(0:temp_time2-1,windowed_Tx_data);
grid on
ylabel('Amplitude (volts)')
xlabel('Time (samples)')
title('OFDM windowed_Tx_data')

%% PAPR ��Power Complementary Cumulative Distribution Function�����ʻ����ۼƻ��ֺ����� Derivation for OFDM
PAPR = zeros(1,IFFT_bin_length+GI+GIP);
PAPR_windowed = zeros(1,IFFT_bin_length+GI+GIP);
for i = 1:IFFT_bin_length+GI+GIP   %deriving the papr of all subcarriers
    PAPR(i)= 10*log10(max(abs(time_wave_matrix_cp(:,i)).^2)/mean(abs(time_wave_matrix_cp(:,i))).^2);
    PAPR_windowed(i)= 10*log10(max(abs(windowed_time_wave_matrix_cp(:,i)).^2)/mean(abs(windowed_time_wave_matrix_cp(:,i))).^2);
end
[Y1,X1] = hist(PAPR,200);%[������Ԫ�ظ���������������ֵ]=hist[��ͳ�Ƶ����ݣ������ݻ��ֵ��������]
[Y2,X2] = hist(PAPR_windowed,200);
i = 1:IFFT_bin_length+GI+GIP;
%δ�Ӵ�
figure(11)
subplot(2,1,1);
plot(i,PAPR,'-b', 'LineWidth',1.2);
title('PAPR of each subcarrier','FontSize',15)
xlabel('subcarrier','FontSize',15)
ylabel('PAPR, dB','FontSize',15)
subplot(2,1,2);
plot(X1,1-cumsum(Y1)/max(cumsum(Y1)),'-b', 'LineWidth',1.2);%cumsum()���ۼƺ�1��2��3->1,3,6
title('Power CCDF of OFDM','FontSize',15)
xlabel('PAPR, dB','FontSize',15)
ylabel('Probability','FontSize',15)
%�Ӵ�������׺����ǰ׺�ص�
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

%% =================δ�Ӵ���ͼ==================================
symbols_per_average = ceil(symbols_per_carrier/5);% Ϊʲô/5��
avg_temp_time = (IFFT_bin_length+GI+GIP)*symbols_per_average;%Ϊʲôȡ3��ofdm������fft��
averages = floor(temp_time1/avg_temp_time);
average_fft(1:avg_temp_time) = 0;
for a = 0:(averages-1)
    subset_ofdm = Tx_data_withoutwindow (((a*avg_temp_time)+1):((a+1)*avg_temp_time));
    subset_ofdm_f = abs(fft(subset_ofdm));
    average_fft = average_fft + (subset_ofdm_f/averages);%Ƶ��/���������ʱ��С�飿
end
average_fft_log = 20*log10(average_fft);
figure (6)
subplot(2,1,1);
plot((0:(avg_temp_time-1))/avg_temp_time, average_fft_log)% 0/avg_temp_time  :  (avg_temp_time-1)/avg_temp_time
hold on
plot(0:1/IFFT_bin_length:1, -35, 'rd')%-35���Ǵ������ĳ�����
grid on
axis([0 0.5 -40 max(average_fft_log)])
ylabel('Magnitude (dB)')
xlabel('Normalized Frequency (0.5 = fs/2)')
title('OFDM Signal Spectrum without windowing')
%% ===============�Ӵ���ͼ=================================
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
%% �ϱ�Ƶ



%% ====================�������============================================
Tx_signal_power = var(windowed_Tx_data);%�õ��������ݹ���
linear_SNR=10^(SNR/10);%�����ת����int��
noise_sigma=Tx_signal_power/linear_SNR;%��������
noise_scale_factor = sqrt(noise_sigma);%������׼��
noise=randn(1,((symbols_per_carrier)*(IFFT_bin_length+GI))+GIP)*noise_scale_factor;%���������randn��ֵΪ0�������^2 = 1����׼��� = 1����̬�ֲ����������D��cX��=c^2 * D��X��
 
%noise=wgn(1,length(windowed_Tx_data),noise_sigma,'complex');
 
Rx_data=windowed_Tx_data+noise;%��������
%% �±�Ƶ



%% �����Ƕྶ������ͬ����AWGN
%% =====================ѭ��ǰ��׺ȥ��==========================================
Rx_data_matrix=zeros(symbols_per_carrier,IFFT_bin_length+GI+GIP);
for i=1:symbols_per_carrier
    Rx_data_matrix(i,:)=Rx_data(1,(i-1)*(IFFT_bin_length+GI)+1:i*(IFFT_bin_length+GI)+GIP);%�������źŷ�Ϊ12������
end
Rx_data_complex_matrix=Rx_data_matrix(:,GI+1:IFFT_bin_length+GI);%ȥ��CP��CPI


%% =================FFT=================================
Y1=fft(Rx_data_complex_matrix,IFFT_bin_length,2);
% Ƶ����Ϣ
Rx_carriers=Y1(:,carriers);%��ȡ������
Rx_phase =angle(Rx_carriers);%�����λ��Ϣ
Rx_mag = abs(Rx_carriers);%��÷�����Ϣ
figure(7);
polar(Rx_phase, Rx_mag,'bd');%�������ݵ���λ�ͷ�����Ϣ��ͼ
title('Phase and mapulitude of Rx_data');
%======================================================================
% ��ͼ��ʽ1
[M, N]=pol2cart(Rx_phase, Rx_mag); %������ת�ѿ�������
Rx_complex_carrier_matrix = complex(M, N);%�õ����ź�
figure(8);
plot(Rx_complex_carrier_matrix,'*r');%�����źŵ�����ͼ
title('star map of Rx_data');
axis([-4, 4, -4, 4]);
grid on
% ��ͼ��ʽ2
figure(9);
plot(Rx_carriers,'*r');%�����źŵ�����ͼ
title('star map of Rx_data');
axis([-4, 4, -4, 4]);
grid on

%% ********ģ��ʱ�ӡ��ٶ���Ϣ***********************************************************
% ������һ���������䣬��û�����±�Ƶ
%*******************������ٶȲ�������*******************
% ������Ŀ��
V = -400; % m/s
R = 29.25; % m
theta = pi/6;% rad
RX_num = 8;%RX����ͨ����
Angle_fft_num = RX_num;%����άfft����
lambda = c/f_c;   %ofdm�źŲ���
d = lambda/2;%������Ԫ���
%*******************����Kr �� Kd ����(����strum������)*******************
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
% Ƶ�����ʱ�Ӷ�������Ϣ��Rx_complex_carrier_matrix��Ƶ����ʽ
gg=kd' *  kr;
Rx_complex_carrier_matrix_radar = 1 * Rx_complex_carrier_matrix .* (kd' *  kr);%ʱ�Ӷ��������fft����Ӱ�죬ֱ�ӳ���Ƶ��Ҳ��
% ���������߽��վ���
multi_Rx_complex_carrier_matrix_radar = repmat(zeros(size(Rx_complex_carrier_matrix_radar)),[1 1 RX_num]);
for page=1:RX_num
    multi_Rx_complex_carrier_matrix_radar(:,:,page) = Rx_complex_carrier_matrix_radar * ka(page);
end
%% ********OFDMͨ���źŴ���***********************************************************
% ����������߽��յ���ֻ�����˵����߽���
%====================4qam����==================================================
Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
Rx_decoded_binary_symbols=demoduqam4(Rx_serial_complex_symbols);

% %====================8qam����==================================================
% Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
% Rx_decoded_binary_symbols=demoduqam8(Rx_serial_complex_symbols);

% %====================16qam����==================================================
% Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
% Rx_decoded_binary_symbols=demoduqam16(Rx_serial_complex_symbols);

% %====================32qam����==================================================
% Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
% Rx_decoded_binary_symbols=demoduqam32(Rx_serial_complex_symbols);

% %====================64qam����==================================================
% Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix',size(Rx_complex_carrier_matrix, 1)*size(Rx_complex_carrier_matrix,2),1)';
% Rx_decoded_binary_symbols=demoduqam64(Rx_serial_complex_symbols);


%============================================================
baseband_in = Rx_decoded_binary_symbols;
 
figure(9);
subplot(2,1,1);
stem(baseband_out(1:100));
subplot(2,1,2);
stem(baseband_in(1:100));
title('sending beta�� 1-200');
%================�����������=============================================
bit_errors=find(baseband_in ~=baseband_out);
bit_error_count = size(bit_errors, 2);
ber=bit_error_count/baseband_out_length;

%% ********OFDM�״��źŴ���***********************************************************
%���
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

% ����
Velocity_fft=zeros(size(multi_Rx_complex_carrier_matrix_radar));
for j=1:RX_num
    for i=1:carrier_count
        r_ifft_step = Range_ifft(:,i,j);
        v_fft_step = fftshift(fft(r_ifft_step));
        Velocity_fft(:,i,j)=v_fft_step;
    end
end

% ���
Angle_fft=zeros(symbols_per_carrier, carrier_count, Angle_fft_num);
for j=1:symbols_per_carrier
    for i=1:carrier_count
        v_fft_step = Velocity_fft(j, i, :);
        a_fft_step = fftshift(fft(v_fft_step, Angle_fft_num));
        Angle_fft(j, i, :)=a_fft_step;
    end
end

% �����ֵλ��,������Ŀ���˶�״̬
Angle_profile=abs(Angle_fft);
peak=max(Angle_profile(:));%��Ŀ��ȡ���ֵ
[index_v,index_r,index_a]=ind2sub(size(Angle_profile),find(Angle_profile==peak));

M_R = ((index_r-1) / carrier_count) * (c / 2 / delta_f)
N_V = ((index_v-symbols_per_carrier/2-1) / symbols_per_carrier) * (c / 2 / f_c/T_OFDM)
T_A = asin((index_a-Angle_fft_num/2-1) / Angle_fft_num*(lambda/d))*(180/pi)

%�ٶ�-��������ͼ
b=-symbols_per_carrier/2:1:symbols_per_carrier/2-1;
a=1:1:carrier_count;
figure
[A,B] = meshgrid(a.*(c / 2 / delta_f)/carrier_count,b.*(c / 2 / f_c/T_OFDM)/symbols_per_carrier);
mesh(A,B,abs(Velocity_fft(:,:,1)));
xlabel('����/m');ylabel('�ٶȣ�m/s��');zlabel('�źŷ�ֵ');
title('2άFFT������ά��ͼ');

%���뷽λ����ͼ
b=1:1:carrier_count;
a=-Angle_fft_num/2:1:Angle_fft_num/2-1;
figure;
Angle_profile_temp = reshape(sum(Angle_profile, 1),carrier_count,Angle_fft_num);
[X,Y]=meshgrid(asin(a .* (lambda/d) / Angle_fft_num)*180/pi, b.*(c / 2 / delta_f)/carrier_count);
mesh(X,Y,(abs(Angle_profile_temp))); 
xlabel('theta/��');ylabel('����(m)');zlabel('�źŷ�ֵ');
title('2άFFT������ά��ͼ');

% %���
% figure(10)
% div_IFFT = div_IFFT / symbols_per_carrier;
% plot(abs(div_IFFT));
% text(floor(index),abs(div_IFFT(round(index))),'o','color','r');  %��ǳ���ֵ��
% text(index + 5,abs(div_IFFT(round(index))) + 0.02,['(',num2str(index),',',num2str(abs(div_IFFT(round(index)))),')'],'color','r');  %��ǳ���ֵ��
% xlabel('index');
% ylabel('Mag');
% title('OFDM radar ranging');
