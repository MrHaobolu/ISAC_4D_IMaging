function [threshold_matrix,target_index_processed,detect_matrix_abs] = CA_CFAR(detect_matrix)
%本函数用于针对A_2域进行恒虚警检测
%   本函数对角度FFT结果进行恒虚警检测
threshold_matrix = zeros(size(detect_matrix));
target_index = [];
target_index_processed = [];
% 关键参数
window_size = 9;
guard_window_size = 3;
N = window_size - 1;
Pfa = 1e-3; % 虚警概率，经典值

threshold_adjust = 60000;
point_richness = 3;% 可以根据角度分辨率调整

detect_matrix_abs = abs(detect_matrix) ;%.* abs(detect_matrix);% 平方律检测器
for i=1:size(detect_matrix, 1)
    for j=1:size(detect_matrix, 2)
        CUT = detect_matrix_abs(i, j);
        detect_window = zeros(window_size, window_size);
        % 检测窗口行索引
        if i<(N/2+1)
            row_index = 1:window_size;
        elseif (size(detect_matrix, 1)-i)<N/2
            row_index = size(detect_matrix, 1) - window_size + 1 : size(detect_matrix, 1);
        else
            row_index = i-N/2 : i + N/2;
        end
        % 检测窗口列索引
        if j<(N/2+1)
            col_index = 1:window_size;
        elseif (size(detect_matrix, 2)-j)<N/2
            col_index = size(detect_matrix, 2) - window_size + 1 : size(detect_matrix, 2);
        else
            col_index = j-N/2 : j + N/2;
        end
        % 检测窗口赋值
        for m=1:length(row_index)
            for n=1:length(col_index)
                detect_window(m, n) = detect_matrix_abs(row_index(m), col_index(n));
                % CUT单元在检测窗口中的位置
                if row_index(m)==i && col_index(n)==j
                    CUT_pos_row = m;
                    CUT_pos_col = n;
                end
            end
        end
        % 保护单元在检测窗口中的行列索引
        row_start = CUT_pos_row - (guard_window_size - 1) / 2;
        row_end = CUT_pos_row + (guard_window_size - 1) / 2;
        col_start = CUT_pos_col - (guard_window_size - 1) / 2;
        col_end = CUT_pos_col + (guard_window_size - 1) / 2;
        if row_start < 1
            row_start = 1;
        end
        if row_end > window_size
            row_end = window_size;
        end
        if col_start < 1
            col_start = 1;
        end
        if col_end > window_size
            col_end = window_size;
        end
        guard_row = row_start:row_end;
        guard_col = col_start:col_end;
        % 将保护单元以及CUT置零
        for index_guard_row=1:length(guard_row)
            for index_guard_col=1:length(guard_col)
                detect_window(guard_row(index_guard_row), guard_col(index_guard_col)) = 0;
            end
        end
        
        % CA_CFAR
        noise_power = sum(detect_window(:)) / (window_size^2-length(guard_row)*length(guard_col));
        % 检测门限
        K_factor = Pfa^(-1 / (window_size^2-length(guard_row)*length(guard_col))) - 1; % 检测阈值因子
        % K_factor = Pfa^(-1 / (window_size-length(guard_row))) - 1; % 检测阈值因子
        threshold_matrix(i ,j) = noise_power * K_factor + threshold_adjust;
        if CUT>threshold_matrix(i ,j) 
            target_index = [target_index ;i j CUT];
        end
    end
end
% 控制返回点数
if size(target_index, 1)>point_richness
    CUT_matrix = target_index(:, 3);
    [~, IX]=sort(CUT_matrix,1,"descend");
    for i=1:point_richness
        target_index_processed = [target_index_processed;target_index(IX(i, 1), 1) target_index(IX(i, 1), 2)];
    end
elseif size(target_index, 1)==0
    target_index_processed = target_index;
else
    target_index_processed = target_index(:, 1:2);
end

end