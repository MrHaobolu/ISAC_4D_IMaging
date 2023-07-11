%% ���������һ��ofdm��֡���е��Ƴ���
clc;
clear;
close all;

ref_space = 4;%ÿ4��ʱ϶����һ�鵼Ƶ
slot_num=16;%ʱ϶��
IFFT_length=2048;%���ز���Դ��
carrier_count=200;%���ز���%%%%%%
ref_carrier_count = IFFT_length/4;%��ƵƵ��ռ��,comb4
comb_num = 4;
symbols_per_carrier=224;%OFDM������,�����ز�����Ϊ240kHz�������£�ÿ����֡����16��ʱ϶��16*14��ofdm���ţ�ʹ��һ����֡
ref_symbol_count = (slot_num/ref_space) * 12;%��Ƶʱ��ռ��,ÿ4��ʱ϶����һ��PRS��12�����ţ�

bits_per_symbol=2;%ÿ�����Ʒ��ŵı�������4QAM��Ӧ�ľ���2��
% bits_per_symbol=3;%ÿ�����Ʒ��ŵı�������8QAM��Ӧ�ľ���3��
% bits_per_symbol=4;%ÿ�����Ʒ��ŵı�������16QAM��Ӧ�ľ���4��
% bits_per_symbol=5;%ÿ�����Ʒ��ŵı�������32QAM��Ӧ�ľ���5��
% bits_per_symbol=6;%ÿ�����Ʒ��ŵı�������64QAM��Ӧ�ľ���6��


PrefixRatio=1/4;%ѭ��ǰ׺���� 1/6~1/4
GI=PrefixRatio*IFFT_length ;%ѭ��ǰ׺�ĳ���
beta=1/32;%ѭ����׺����
GIP=beta*(IFFT_length+GI);%ѭ����׺����
SNR=20; %����ȣ���λ��dB

%% ������������
c = 3*10^8;  % ��Ų������ٶȣ� ��λm/s
delta_f = 240*10^3;  % �ز��������λhz
f_c = 70*10^9; % �ź�����Ƶƫ, ��λhz, 25GHz,n258Ƶ��
%% ================������������===================================
disp('���ڹ�����������ͨ����Ϣ��PRS����......');
baseband_out_length = IFFT_length * symbols_per_carrier * bits_per_symbol;%���ͱ��س���
PSF_num = ref_carrier_count * ref_symbol_count * bits_per_symbol;% �ο��źű��س���
Info_num = baseband_out_length - PSF_num;%ͨ���źű��س���
% �ο��ź�REƵ������
carriers_f = zeros(comb_num, ref_carrier_count);
Info_carriers_f_1 = zeros(comb_num, IFFT_length - ref_carrier_count);%��һ��ͨ����ϢƵ������
f_offset = [0,2,1,3];
for i=1:comb_num
    offset = f_offset(i);
    for j=1:ref_carrier_count
        carriers_f(i,j) = comb_num*(j-1)+1 + offset;
    end
    Info_carriers_f_1(i, :) = setdiff((1:IFFT_length), carriers_f(i,:));
end
Info_carriers_f_1_full = repmat(Info_carriers_f_1, (12/comb_num) * ref_space, 1);%��һ��ͨ����ϢƵ������
carriers_f_full = repmat(carriers_f, (12/comb_num) * ref_space, 1);
% �ο��ź�REʱ������
symbols_t = zeros(1,ref_symbol_count);
for i=1:slot_num/ref_space
    symbols_t(1,(i-1)*12+1:i*12) = 14*ref_space*(i-1)+2 : 14*ref_space*(i-1)+13;
end
Info_symbol_t_1 = symbols_t;%��һ��ͨ����Ϣʱ������

Info_carriers_f_2 = repmat((1:IFFT_length), (symbols_per_carrier - ref_symbol_count), 1);
Info_symbol_t_2 = setdiff((1:slot_num*14), symbols_t);% �ڶ���ͨ����Դʱ������

% �ο��ź�����
ref_t_f_index = zeros(size(symbols_t,2), size(carriers_f_full,2), 2);
ref_t_f_index(:, :, 1)=repmat(symbols_t', 1, ref_carrier_count);
ref_t_f_index(:, :, 2)=carriers_f_full;

% ͨ���ź�����
% class 1
Info_t_f_index_1 = zeros(size(Info_symbol_t_1,2), size(Info_carriers_f_1_full,2), 2);
Info_t_f_index_1(:, :, 1)=repmat(Info_symbol_t_1', 1, (IFFT_length - ref_carrier_count));
Info_t_f_index_1(:, :, 2)=Info_carriers_f_1_full;
%class 2
Info_t_f_index_2 = zeros(size(Info_symbol_t_2,2), size(Info_carriers_f_2,2), 2);
Info_t_f_index_2(:, :, 1)=repmat(Info_symbol_t_2', 1, IFFT_length);
Info_t_f_index_2(:, :, 2)=Info_carriers_f_2;
disp('��������ͨ����Ϣ��PRS����������ϣ�');
%% ***************ģ�����ͨ����Ϣ***************
% baseband_out=zeros(1,baseband_out_length);%��0����

%% ***************ģ�����ͨ����Ϣ***************
% baseband_out=ones(1,baseband_out_length);%��1����

%% ***************ģ�����ͨ����Ϣ***************
% rand( 'twister',0); %������ӣ�������ÿ�����ɵ�������й��ɣ�������仰ÿ�����ɵ����������һ���Ĺ���
% baseband_out=round(rand(1,baseband_out_length));%�����������ģ�������ͨ�����ݣ�
%% ***************m����ģ������ͨ����Ϣ***************
disp('����ʹ��m�������ͨ����Ϣ���ݱ���......');
Tx_matric = zeros(IFFT_length*bits_per_symbol,symbols_per_carrier);
for i = 1:symbols_per_carrier
    Order_number=12; %m���еĽ�������10��m���г���Ϊ2^10 - 1
    mg = zeros(IFFT_length*bits_per_symbol,1);
%    ����m���б�Դ����ʽ��ϵ����������primpoly(Order_number,'all')�õ���
    tmp = primpoly(Order_number,'all','nodisplay'); %�������п��е�m���еı�Դ����ʽϵ��������Խ����Խ��(������ϵ��������)
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
    mg(1:IFFT_length*bits_per_symbol-1,1) = tmp(1:IFFT_length*bits_per_symbol-1);
    Tx_matric(:,i) = mg;
end
Tx_matric = Tx_matric';
disp('ͨ����Ϣ���ݱ��������ϣ�');
%% ****************gold����ģ��PRS��Ϣ*****************************
disp('����ʹ��gold�������PRS���ݱ���......');
    for m=1:ref_symbol_count
        n_slot=ceil(symbols_t(m)/14)-1;
        seq=goldseq(n_slot,symbols_t(m)-1);%α�������
        for k=1:ref_carrier_count
           Tx_matric(ref_t_f_index(m,k,1), 2*ref_t_f_index(m,k,2)-1) = seq(2*ref_t_f_index(m,k,2)-1);
           Tx_matric(ref_t_f_index(m,k,1), 2*ref_t_f_index(m,k,2)) = seq(2*ref_t_f_index(m,k,2));
        end
    end

baseband_out = reshape(Tx_matric',1,baseband_out_length); 
disp('PRS���ݱ���������......');
%% *************************���ݵ���****************************
% ==============4QAM����====================================
disp('���ڽ������ݵ��ƣ���ʱ�ϳ�,���Ժ�......');
complex_carrier_matrix=qam4(baseband_out);
complex_carrier_matrix=reshape(complex_carrier_matrix',IFFT_length, symbols_per_carrier)';%symbols_per_carrier*carrier_count
 
figure
plot(complex_carrier_matrix,'*r');
title('star map of Tx_data');
axis([-2, 2, -2, 2]);
grid on
% % ==============8QAM����====================================
% complex_carrier_matrix=qam8(baseband_out);
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-2, 2, -2, 2]);
% grid on


% % ==============16QAM����====================================
% complex_carrier_matrix=qam16(baseband_out);
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-4, 4, -4, 4]);
% grid on

% % ==============32QAM����====================================
% complex_carrier_matrix=qam32(baseband_out);
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-8, 8, -8, 8]);
% grid on

% % ==============64QAM����====================================
% complex_carrier_matrix=qam64(baseband_out);
% complex_carrier_matrix=reshape(complex_carrier_matrix',carrier_count,symbols_per_carrier)';%symbols_per_carrier*carrier_count
%  
% figure(1);
% plot(complex_carrier_matrix,'*r');
% title('star map of Tx_data');
% axis([-8, 8, -8, 8]);
% grid on

disp('������ϣ�');
%% =================IFFT===========================
disp('����ʹ��IFFT����ʱ��OFDM����......');
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
disp('ʱ��OFDM����������ϣ�');
%% =====================��ѭ��ǰ׺CP�ͺ�׺=========================
disp('�������ѭ��ǰ׺��׺......');
time_wave_matrix_cp=zeros(symbols_per_carrier,IFFT_length+GI+GIP);%GI=512,GIP=80
for k=1:symbols_per_carrier    %224
    time_wave_matrix_cp(k,GI+1:GI+IFFT_length)=signal_after_IFFT(k,:);
    time_wave_matrix_cp(k,1:GI)=signal_after_IFFT(k,(IFFT_length-GI+1):IFFT_length);%��ѭ��ǰ׺CP,ǰ׺�Ƿ��ź���Ĳ���
    time_wave_matrix_cp(k,(IFFT_length+GI+1):(IFFT_length+GI+GIP))=signal_after_IFFT(k,1:GIP);%��ѭ����׺����׺�Ƿ���ǰ��Ĳ���
end
subplot(3,1,2);
plot(0:length(time_wave_matrix_cp)-1,time_wave_matrix_cp(2,:));
axis([0, 3000, -0.2, 0.2]);
grid on;  
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal with CP, One Symbol Period');
disp('ѭ��ǰ׺��׺�����ϣ�');
%% ***************OFDM���żӴ�����******************
disp('����ʱ��Ӵ������в���ת��......');
windowed_time_wave_matrix_cp=zeros(symbols_per_carrier,IFFT_length+GI+GIP);
for i = 1:symbols_per_carrier %224
    windowed_time_wave_matrix_cp(i,:) = time_wave_matrix_cp(i,:).*rcoswindow(beta,IFFT_length+GI)';%�����ҹ���ϵ��=ѭ����׺����
end  
subplot(3,1,3);
plot(0:IFFT_length-1+GI+GIP,windowed_time_wave_matrix_cp(2,:));
axis([0, 3000, -0.2, 0.2]);
grid on;
ylabel('Amplitude');
xlabel('Time');
title('OFDM Time Signal Apply a Window , One Symbol Period');
  
% ����ת�����Ӵ���ofdm���Ŵ���ʱ��ǰ���ŵĺ�׺����һ�����ŵ�ǰ׺�غ�
windowed_Tx_data=zeros(1,symbols_per_carrier*(IFFT_length+GI)+GIP);
windowed_Tx_data(1:IFFT_length+GI+GIP)=windowed_time_wave_matrix_cp(1,:);
for i = 1:symbols_per_carrier-1 
    windowed_Tx_data((IFFT_length+GI)*i+1:(IFFT_length+GI)*(i+1)+GIP)=windowed_time_wave_matrix_cp(i+1,:);%�Ӵ����Һ�׺��ǰ׺�ص�
end
%�Ӵ���׺��ǰ׺�ص�
figure
temp_time2 =symbols_per_carrier*(IFFT_length+GI)+GIP;
plot(0:temp_time2-1,windowed_Tx_data);
grid on
ylabel('Amplitude (volts)')
xlabel('Time (samples)')
title('OFDM windowed_Tx_data')
disp('���з����ź����ɳɹ���');
%% �ϱ�Ƶ

%% ====================�������============================================
disp('�������AWGN......');
Tx_signal_power = var(windowed_Tx_data);%�õ��������ݹ���
linear_SNR=10^(SNR/10);%�����ת����int��
noise_sigma=Tx_signal_power/linear_SNR;%��������
noise_scale_factor = sqrt(noise_sigma);%������׼��
noise=randn(1,((symbols_per_carrier)*(IFFT_length+GI))+GIP)*noise_scale_factor;%���������randn��ֵΪ0�������^2 = 1����׼��� = 1����̬�ֲ����������D��cX��=c^2 * D��X��
%noise=wgn(1,length(windowed_Tx_data),noise_sigma,'complex');
Rx_data=windowed_Tx_data+noise;%��������
disp('AWGN�����ϣ�');
%% �±�Ƶ

%% �����Ƕྶ������ͬ����AWGN
%% =====================����ת����ѭ��ǰ��׺ȥ��==========================================
disp('���ڽ��д���ת����ѭ��ǰ��׺����......');
Rx_data_matrix=zeros(symbols_per_carrier,IFFT_length+GI+GIP);
%����ת��
for i=1:symbols_per_carrier
    Rx_data_matrix(i,:)=Rx_data(1,(i-1)*(IFFT_length+GI)+1:i*(IFFT_length+GI)+GIP);%�������źŷ�Ϊ224������
end
%ѭ��ǰ��׺ȥ��
Rx_data_complex_matrix=Rx_data_matrix(:,GI+1:IFFT_length+GI);%ȥ��CP��CPI
disp('�����źŻָ��ɹ���');
%% =================FFT=================================
disp('����ʹ��FFT�ָ�Ƶ����Ϣ......');
Y1=fft(Rx_data_complex_matrix,IFFT_length,2);
% Ƶ����Ϣ
Rx_carriers=Y1;%��ȡ������
Rx_phase =angle(Rx_carriers);%�����λ��Ϣ
Rx_mag = abs(Rx_carriers);%��÷�����Ϣ
figure
polar(Rx_phase, Rx_mag,'bd');%�������ݵ���λ�ͷ�����Ϣ��ͼ
title('Phase and mapulitude of Rx_data');
%======================================================================
% ��ͼ��ʽ1
[M, N]=pol2cart(Rx_phase, Rx_mag); %������ת�ѿ�������
Rx_complex_carrier_matrix = complex(M, N);%�õ����ź�
figure
plot(Rx_complex_carrier_matrix,'*r');%�����źŵ�����ͼ
title('star map of Rx_data');
axis([-4, 4, -4, 4]);
grid on
% % ��ͼ��ʽ2
% figure
% plot(Rx_carriers,'*r');%�����źŵ�����ͼ
% title('star map of Rx_data');
% axis([-4, 4, -4, 4]);
% grid on
disp('Ƶ����Ϣ�ָ��ɹ���');
%% ********OFDMͨ���źŴ���***********************************************************
disp('���ڽ����ȡԭʼ���ݱ���......');
% ����������߽��յ���ֻ�����˵����߽���
%====================4qam����==================================================
Rx_serial_complex_symbols = reshape(Rx_complex_carrier_matrix', 1, size(Rx_complex_carrier_matrix,1)*size(Rx_complex_carrier_matrix,2))';
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
disp('�����ϣ�');
figure
subplot(2,1,1);
stem(baseband_out(1:100));
subplot(2,1,2);
stem(baseband_in(1:100));
title('sending beta�� 1-200');
%================�����������=============================================
bit_errors=find(baseband_in ~=baseband_out);
bit_error_count = size(bit_errors, 2);
disp('������Ϊ��');
BER=bit_error_count/baseband_out_length
%% ********ģ��ʱ�ӡ��ٶ���Ϣ***************************************
% ������һ���������䣬��û�����±�Ƶ
%*******************������ٶȲ�������*******************
disp('����ģ���ٶ�ʱ���뷽λ��Ϣ......');
% ɢ�价��
environment_point=environment();
% ��վ����λ��
base_pos=[14,100,20];
% ����ɢ�����Ϣ
point_info=zeros(size(environment_point,1),4);
base_pos_full=repmat(base_pos,size(environment_point,1),1);
% ����
R_info=((environment_point(:,1)-base_pos_full(:,1)).^2+(environment_point(:,2)-base_pos_full(:,2)).^2+(environment_point(:,3)-base_pos_full(:,3)).^2).^(1/2);
% �ٶ�
V_info=environment_point(:,4);
% �Ƕ�
xoy_dis=((environment_point(:,1)-base_pos_full(:,1)).^2+(environment_point(:,2)-base_pos_full(:,2)).^2).^(1/2);
A1_info=acos((base_pos_full(:,1)-environment_point(:,1))./xoy_dis);
A2_info=acos((base_pos_full(:,3)-environment_point(:,3))./R_info);
point_info(:,1)=R_info;
point_info(:,2)=V_info;
point_info(:,3)=A1_info;
point_info(:,4)=A2_info;
disp('�ٶȡ�ʱ�ӡ���λ��Ϣģ����ϣ�');
%% �ز��źŹ������ٶȾ��룩
disp('�������ɵ����߻ز��źţ����ܺ�ʱ�ϳ������Ժ�......');
% ��������������У�����4��64*64���գ������������Ϊ256*256
M = 32;         %x������Դ��
N = 32;        %y������Դ��
lambda = c/f_c;   %ofdm�źŲ���
d = lambda/2;%������Ԫ���
%*******************����kr��kd��ka ����*******************
T_OFDM = 1/delta_f * (1 + PrefixRatio);
% ���������߽��վ���
% multi_Rx_complex_carrier_matrix_radar_RX1 = zeros(size(Rx_complex_carrier_matrix));
multi_Rx_complex_carrier_matrix_radar = zeros(symbols_per_carrier, IFFT_length, M, N);

win = waitbar(0, '�ز�������...');
tCount1 = 0;
for tgt_index = 1:size(point_info, 1)
    % ��ʱ��ʼ��
    t00 = tic;
    % ��ȡ��Ŀ����Ϣ
    R = point_info(tgt_index, 1); % Ŀ�����
    V = point_info(tgt_index, 2); % Ŀ���ٶ�
    theta = point_info(tgt_index, 3); % Ŀ�귽λ��
    faii = point_info(tgt_index,4); % Ŀ�긩����

    % ��Ŀ�������Ϣ
    kr = zeros(1,IFFT_length);
    for k = 1:IFFT_length
        kr(k) = exp(-1i * 2 * pi * (k-1) * delta_f * 2 * R / c);
    end
    % ��Ŀ���ٶ���Ϣ
    kd = zeros(1,symbols_per_carrier);
    for k = 1:symbols_per_carrier
        kd(k) = exp(1i * 2 * pi * T_OFDM * (k-1) * 2 * V * f_c / c);
    end
    
    % Ƶ�����ʱ�Ӷ�������Ϣ��Rx_complex_carrier_matrix��Ƶ����ʽ
    Rx_complex_carrier_matrix_radar = 1 * Rx_complex_carrier_matrix .* (kd' *  kr);%ʱ�Ӷ��������fft����Ӱ�죬ֱ�ӳ���Ƶ��Ҳ��
    
    %�����ߵ�Ŀ��Ƕ���Ϣ
    ka=zeros(M,N);
    for index_x=1:M
        for index_y=1:N
              % ���̲�Ϊ-���Ҿ����ϵ�����仯
            if theta>(90*pi/180)
                if faii<=(90*pi/180)
                    r=(index_x-1)*d*cos(pi-theta)-(index_y-1)*d*sin(pi-theta);%���̲�ˮƽͶӰ
                    ka(index_x,index_y)=exp(-1j*2*pi*r*cos(faii)/lambda); 
                end
                if faii>(90*pi/180)
                    r=(index_x-1)*d*cos(pi-theta)+(index_y-1)*d*sin(pi-theta);%���̲�ˮƽͶӰ
                    ka(index_x,index_y)=exp(-1j*2*pi*r*(cos(pi-faii))/lambda); 
                end
            end
              %���̲�Ϊ+
            if theta<=(90*pi/180)
                
                if faii<=(90*pi/180)
                    r=(index_x-1)*d*cos(theta)+(index_y-1)*d*sin(theta);%���̲�ˮƽͶӰ
                    ka(index_x,index_y)=exp(1j*2*pi*r*cos(faii)/lambda);
                end
                if faii>(90*pi/180)
                    r=(index_x-1)*d*cos(theta)-(index_y-1)*d*sin(theta);%���̲�ˮƽͶӰ
                    ka(index_x,index_y)=exp(1j*2*pi*r*(cos(pi-faii))/lambda); 
                end
                 
            end
            multi_Rx_complex_carrier_matrix_radar(:,:,index_x,index_y) = multi_Rx_complex_carrier_matrix_radar(:,:,index_x,index_y) + Rx_complex_carrier_matrix_radar * ka(index_x,index_y);
        end 
    end
    % multi_Rx_complex_carrier_matrix_radar_RX1 = multi_Rx_complex_carrier_matrix_radar_RX1 + Rx_complex_carrier_matrix_radar * ka(1,1);
    % ʣ��ʱ��Ԥ��
    tCount1 = tCount1 + toc(t00);
    t_step = tCount1/tgt_index;
    t_res = (size(point_info, 1) - tgt_index) * t_step;
    str=['ʣ������ʱ�䣺',num2str(t_res/60),'min'];
    waitbar(tgt_index/size(point_info, 1), win, str)
end
close(win);
disp('�����߻ز��ź�������ϣ�')
%% ********OFDM�״��źŴ������ٲ�ࣩ***********************************************************
disp('��ʼ���ٲ��......')
% %���
% Range_ifft=zeros(size(multi_Rx_complex_carrier_matrix_radar(:, :, 1, 1)));
% % for i=1:M
% %     for j=1:N
%         div_page = multi_Rx_complex_carrier_matrix_radar(:, :, 1, 1) ./ complex_carrier_matrix;
%         for k=1:symbols_per_carrier
%             div_ifft_step = div_page(k,:);
%             r_ifft = ifft(div_ifft_step);
%             Range_ifft(k,:)=r_ifft;
%         end
% %     end
% % end
% 
% % ����
% Velocity_fft=zeros(size(multi_Rx_complex_carrier_matrix_radar(:, :, 1, 1)));
% % for i=1:M
% %     for j=1:N
%         for k=1:IFFT_length
%             r_ifft_step = Range_ifft(:,k);
%             v_fft_step = fftshift(fft(r_ifft_step));
%             Velocity_fft(:,k)=v_fft_step;
%         end
% %     end
% % end

% ���ٲ��
Velocity_fft = zeros(size(multi_Rx_complex_carrier_matrix_radar));

win = waitbar(0, '����Ϊ�������߻ز�����fft...');
tCount1 = 0;
for i=1:M
    % ��ʱ��ʼ��
    t00 = tic;
    for j=1:N
        div_page = multi_Rx_complex_carrier_matrix_radar(:, :, i, j) ./ complex_carrier_matrix;
        page_ifft = ifft(div_page, IFFT_length, 2);
        page_fft = fftshift(fft(page_ifft, symbols_per_carrier, 1), 1);
        Velocity_fft(:, :, i, j) = page_fft;
    end
    %ʣ��ʱ��Ԥ��
    tCount1 = tCount1 + toc(t00);
    t_step = tCount1/i;
    t_res = (M - i) * t_step;
    str=['ʣ������ʱ�䣺',num2str(t_res/60),'min'];
    waitbar(i/M, win, str)
end
close(win);

% ������Բ��ٲ�������к��龯���
[RD_threshold_matrix,RD_target_index,RD_detect_matrix_abs] = OSCA_CFAR(Velocity_fft(:, :, 1, 1));
disp('���ٲ����ϣ�')
%% ���ٲ������CFAR���޻�ͼ
b=-symbols_per_carrier/2:1:symbols_per_carrier/2-1;
a=1:1:IFFT_length;
figure
[A,B] = meshgrid(a.*(c / 2 / delta_f)/IFFT_length,b.*(c / 2 / f_c/T_OFDM)/symbols_per_carrier);
mesh(A,B,RD_detect_matrix_abs);
axis([50 150 -50 50 0 5e6])
xlabel('����/m');ylabel('�ٶȣ�m/s��');zlabel('�źŷ�ֵ');
title('�ٶȾ���fft���');

b_1=-symbols_per_carrier/2:1:symbols_per_carrier/2-1;
a_1=1:1:IFFT_length;
figure
% hold on
[A_1,B_1] = meshgrid(a_1.*(c / 2 / delta_f)/IFFT_length,b_1.*(c / 2 / f_c/T_OFDM)/symbols_per_carrier);
mesh(A_1,B_1,RD_threshold_matrix);
axis([50 150 -50 50 0 5e6])
xlabel('����/m');ylabel('�ٶȣ�m/s��');zlabel('�źŷ�ֵ');
title('�ٶȾ���fft���޽��');

%% �ز��źŹ������Ƕȣ����ǵ����������϶࣬�����ͨ��OSCA_CFARȷ��Ŀ����ڵ�RE������Զ�ӦRE���ɶ����߽��վ��󣬽�ʡ�ڴ��С����ѹ����ͬʱ���ͽǶȷֱ���Ҫ��
disp('���ڹ��������߽ǶȻز�����......');
Angel_fft_page_num = size(RD_target_index, 1); % ѡȡĿ���Ӧ��RE���в��
Angle_matrix = zeros(M, N, Angel_fft_page_num);

for i=1:Angel_fft_page_num
    Angle_matrix(:, :, i) = Velocity_fft(RD_target_index(i,1), RD_target_index(i,2), :, :);
end
% % win = waitbar(0, '�ز�������...');
% % tCount1 = 0;
% for tgt_index = 1:size(point_info, 1)
%     % % ��ʱ��ʼ��
%     % t00 = tic;
%     % ��ȡ��Ŀ����Ϣ
%     R = point_info(tgt_index, 1); % Ŀ�����
%     V = point_info(tgt_index, 2); % Ŀ���ٶ�
%     theta = point_info(tgt_index, 3); % Ŀ�귽λ��
%     faii = point_info(tgt_index,4); % Ŀ�긩����
% 
%     % ��Ŀ�������Ϣ
%     kr = zeros(1,IFFT_length);
%     for k = 1:IFFT_length
%         kr(k) = exp(-1i * 2 * pi * (k-1) * delta_f * 2 * R / c);
%     end
%     % ��Ŀ���ٶ���Ϣ
%     kd = zeros(1,symbols_per_carrier);
%     for k = 1:symbols_per_carrier
%         kd(k) = exp(1i * 2 * pi * T_OFDM * (k-1) * 2 * V * f_c / c);
%     end
% 
%     % Ƶ�����ʱ�Ӷ�������Ϣ��Rx_complex_carrier_matrix��Ƶ����ʽ
%     Rx_complex_carrier_matrix_radar = 1 * Rx_complex_carrier_matrix .* (kd' *  kr);%ʱ�Ӷ��������fft����Ӱ�죬ֱ�ӳ���Ƶ��Ҳ��
% 
%     %�����ߵ�Ŀ��Ƕ���Ϣ
%     ka=zeros(M,N);
%     for index_x=1:M
%         for index_y=1:N
%               % ���̲�Ϊ-���Ҿ����ϵ�����仯
%             if theta>(90*pi/180)
%                 if faii<=(90*pi/180)
%                     r=(index_x-1)*d*cos(pi-theta)-(index_y-1)*d*sin(pi-theta);%���̲�ˮƽͶӰ
%                     ka(index_x,index_y)=exp(-1j*2*pi*r*cos(faii)/lambda); 
%                 end
%                 if faii>(90*pi/180)
%                     r=(index_x-1)*d*cos(pi-theta)+(index_y-1)*d*sin(pi-theta);%���̲�ˮƽͶӰ
%                     ka(index_x,index_y)=exp(-1j*2*pi*r*(cos(pi-faii))/lambda); 
%                 end
%             end
%               %���̲�Ϊ+
%             if theta<=(90*pi/180)
% 
%                 if faii<=(90*pi/180)
%                     r=(index_x-1)*d*cos(theta)+(index_y-1)*d*sin(theta);%���̲�ˮƽͶӰ
%                     ka(index_x,index_y)=exp(1j*2*pi*r*cos(faii)/lambda);
%                 end
%                 if faii>(90*pi/180)
%                     r=(index_x-1)*d*cos(theta)-(index_y-1)*d*sin(theta);%���̲�ˮƽͶӰ
%                     ka(index_x,index_y)=exp(1j*2*pi*r*(cos(pi-faii))/lambda); 
%                 end
% 
%             end
%         end 
%     end
%     for i=1:Angel_fft_page_num
%         Angle_matrix(:, :, i) = Angle_matrix(:, :, i) + Rx_complex_carrier_matrix_radar(RD_target_index(i, 1), RD_target_index(i, 2)) * ka;
%     end
%     %ʣ��ʱ��Ԥ��
%     % tCount1 = tCount1 + toc(t00);
%     % t_step = tCount1/tgt_index;
%     % t_res = (size(point_info, 1) - tgt_index) * t_step;
%     % str=['ʣ������ʱ�䣺',num2str(t_res/60),'min'];
%     % waitbar(tgt_index/size(point_info, 1), win, str)
% end
% % close(win);
disp('�����߽ǶȻز��ź�������ϣ�');
%% ���
disp('��ʼ���......')
Angle_fft_num = 512; % ����άfft����
Angle_fft_matrix = zeros(Angle_fft_num, Angle_fft_num, Angel_fft_page_num);
Angle_fft_threshold_matrix = zeros(Angle_fft_num, Angle_fft_num, Angel_fft_page_num);
Angle_fft_abs_matrix = zeros(Angle_fft_num, Angle_fft_num, Angel_fft_page_num);
A2_Angle_target_cell = cell(size(RD_target_index, 1), 1);

win = waitbar(0, 'REĿ������...');
tCount1 = 0;
for i=1:Angel_fft_page_num
    % ��ʱ��ʼ��
    t00 = tic;
    W_x=fftshift(fft(Angle_matrix(:, :, i), Angle_fft_num, 2),2);
    Angle_fft_matrix(:, :, i)=fftshift(fft(W_x, Angle_fft_num, 1),1);
    % CA_CFAR���
    [A2_threshold_matrix,A2_target_index,A2_detect_matrix_abs] = CA_CFAR(Angle_fft_matrix(:, :, i));
    Angle_fft_threshold_matrix(:, :, i) = A2_threshold_matrix;
    Angle_fft_abs_matrix(:, :, i) = A2_detect_matrix_abs;
    A2_Angle_target_cell{i, 1} = A2_target_index;

    X = ['��', num2str(i), '��Ŀ��RE������'];
    disp(X);
    %ʣ��ʱ��Ԥ��
    tCount1 = tCount1 + toc(t00);
    t_step = tCount1/i;
    t_res = (Angel_fft_page_num - i) * t_step;
    str=['ʣ������ʱ�䣺',num2str(t_res/60),'min'];
    waitbar(i/Angel_fft_page_num, win, str)
end
close(win);
disp('�����ϣ�');
%% ��ǽ����CFAR���޻�ͼ
figure
a=1:Angle_fft_num;
b=1:Angle_fft_num;
[X,Y]=meshgrid(a,b);
mesh(X,Y,abs(Angle_fft_abs_matrix(:, :, 10)));
axis([200 400 200 300 0 12e5])
title('��λfft���');

figure
a=1:Angle_fft_num;
b=1:Angle_fft_num;
[X,Y]=meshgrid(a,b);
mesh(X,Y,abs(Angle_fft_threshold_matrix(:, :, 10)));
axis([200 400 200 300 0 8e5])
title('��λfft����');
%% �ָ�ԭʼλ����Ϣ
pos_all = [];
angle_all= [];
for i=1:size(RD_target_index,1)
    % �ٶȾ������
    M_R = ((RD_target_index(i, 2)-1) / IFFT_length) * (c / 2 / delta_f);
    N_V = -((RD_target_index(i, 1)-symbols_per_carrier/2-1) / symbols_per_carrier) * (c / 2 / f_c / T_OFDM);
    for j=1:size(A2_Angle_target_cell{i, 1}, 1)
        index_ax = A2_Angle_target_cell{i, 1}(j,1);
        index_ay = A2_Angle_target_cell{i, 1}(j,2);

        % �ռ�Ƕȼ���
        angle_1_estimation = (index_ax-Angle_fft_num/2-1)*lambda/(Angle_fft_num*d);
        angle_2_estimation = (index_ay-Angle_fft_num/2-1)*lambda/(Angle_fft_num*d);
        
        if angle_1_estimation>0 && angle_2_estimation>0
            theta_estimation = abs(atan(angle_2_estimation/angle_1_estimation));
            faii_estimation = acos(angle_2_estimation / sin(theta_estimation));
        elseif angle_1_estimation<=0 && angle_2_estimation>0
            theta_estimation = pi-abs(atan(-angle_2_estimation/angle_1_estimation));
            faii_estimation = acos(angle_2_estimation / sin(theta_estimation));
        elseif angle_1_estimation<=0 && angle_2_estimation<=0
            theta_estimation = pi-abs(atan(-angle_2_estimation/angle_1_estimation));
            faii_estimation = pi-acos(-angle_2_estimation / sin(theta_estimation));
        elseif angle_1_estimation>0 && angle_2_estimation<=0
            theta_estimation = abs(atan(-angle_2_estimation/angle_1_estimation));
            faii_estimation = pi-acos(-angle_2_estimation / sin(theta_estimation));
        end
        angle_all=[angle_all; theta_estimation faii_estimation];

        % �ָ��ѿ�������ϵ��Ϣ
        pos_z = base_pos(3) - M_R * cos(faii_estimation);
        pos_x = base_pos(1) + M_R * sin(faii_estimation) * cos(theta_estimation);
        pos_y = base_pos(2) - M_R * sin(faii_estimation) * sin(theta_estimation);
        % [pos_x, pos_y] = pol2cart(faii_estimation, M_R * sin(theta_estimation));
        % ����λ����Ϣ
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
h = colorbar;%�Ҳ���ɫ��
set(get(h,'label'),'string','�˶��ٶ�');%���Ҳ���ɫ������

%% ��ͬ��ͼ��ʽ��֤��ͼ���̡�ɢ�����Ϣ�������̣�û���⣩
pos_all_true = [];
for i=1:size(A1_info, 1)
    theta_true = point_info(i, 3);
    faii_true = point_info(i, 4);
    R_true = point_info(i, 1);
    v_true = point_info(i, 2);

    % �ָ��ѿ�������ϵ��Ϣ
    pos_z_true = base_pos(3) - R_true * cos(faii_true);
    pos_x_true = base_pos(1) + R_true * sin(faii_true) * cos(theta_true);
    pos_y_true = base_pos(2) - R_true * sin(faii_true) * sin(theta_true);
    % [pos_x, pos_y] = pol2cart(faii_estimation, M_R * sin(theta_estimation));
    % ����λ����Ϣ
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
h = colorbar;%�Ҳ���ɫ��
set(get(h,'label'),'string','�˶��ٶ�');%���Ҳ���ɫ������

