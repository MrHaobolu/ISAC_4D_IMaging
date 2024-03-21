function target_num = WCA_CFAR_1D(detect_list)
%本函数用于针对特征值进行恒虚警检测
% 考虑到特征值数量较少，不适合使用保护单元，否则会引起参考单元数目过少

threshold_list = zeros(size(detect_list));
target_index = [];
target_num = 0;
target_index_processed = [];
% 关键参数
window_size = 5;
% guard_window_size = 0;
N = window_size - 1;
Pfa = 1e-3; % 虚警概率，经典值
K_factor = Pfa^(-1 / (window_size-1)) - 1; % 检测阈值因子

threshold_adjust = 100;
point_richness = 3;% 特征值返回点数控制

detect_list_abs = abs(detect_list);% .* abs(detect_list);% 平方律检测器

for i=1:length(detect_list)
    CUT = detect_list_abs(i);
    detect_window = zeros(1, window_size);
    % 检测窗口索引
    if i<(N/2+1)
        row_index = 1:window_size;
    elseif (length(detect_list)-i)<N/2
        row_index = length(detect_list) - window_size + 1 : length(detect_list);
    else
        row_index = i-N/2 : i + N/2;
    end
    % 检测窗口赋值
    for m=1:length(row_index)
        detect_window(m) = detect_list_abs(row_index(m));
        % CUT单元在检测窗口中的位置
        if row_index(m)==i
            CUT_pos_row = m;
        end
    end
    % % 保护单元在检测窗口中的行列索引
    % row_start = CUT_pos_row - (guard_window_size - 1) / 2;
    % row_end = CUT_pos_row + (guard_window_size - 1) / 2;
    % 
    % if row_start < 1
    %     row_start = 1;
    % end
    % if row_end > window_size
    %     row_end = window_size;
    % end
    % guard_row = row_start:row_end;

    % % 将保护单元以及CUT置零
    % for index_guard_row=1:length(guard_row)
    %     for index_guard_col=1:length(guard_col)
    %         detect_window(guard_row(index_guard_row), guard_col(index_guard_col)) = 0;
    %     end
    % end
    
    % WCA-CFAR
    % 因为特征值从小到大排序，因此考虑右侧比重下降
    factor_left = 0.8;
    factor_right = 1 - factor_left;
    if CUT_pos_row == 1
        avg_left = 0;
        avg_right = sum(detect_window((CUT_pos_row+1):length(detect_window))) / (length(detect_window) - CUT_pos_row);
    elseif CUT_pos_row == length(detect_window)
        avg_left = sum(detect_window(1:(CUT_pos_row-1))) / (CUT_pos_row - 1);
        avg_right = 0;
    else
        avg_left = sum(detect_window(1:(CUT_pos_row-1))) / (CUT_pos_row - 1);
        avg_right = sum(detect_window((CUT_pos_row+1):length(detect_window))) / (length(detect_window) - CUT_pos_row);
    end
    noise_power = avg_left * factor_left + avg_right * factor_right;
    % 检测门限
    threshold_list(i) = noise_power * K_factor + threshold_adjust;
    if CUT>threshold_list(i) 
        target_index = [target_index; i CUT];
        target_num = target_num + 1;
    end
end

% figure
% plot(detect_list);
% hold on
% plot(threshold_list);
% legend('detect_list', 'threshold_list');


% 控制信号子空间点数
if length(target_index)>point_richness
    CUT_matrix = target_index(:, 2);
    [~, IX]=sort(CUT_matrix,1,"descend");
    for i=1:point_richness
        target_index_processed = [target_index_processed;target_index(IX(i, 1), 1)];
    end
    target_num = point_richness;
elseif isempty(target_index)
    target_index_processed = target_index;
else
    target_index_processed = target_index(:, 1);
end



end