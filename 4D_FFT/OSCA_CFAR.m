function [threshold_matrix,target_index,detect_matrix_abs] = OSCA_CFAR(detect_matrix)
%本函数用于针对RD域进行恒虚警检测
%   本函数默认行方向是距离维，列方向为多普勒维，若不符合此描述请先进行转置
threshold_matrix = zeros(size(detect_matrix));
target_index = [];
% 关键参数
window_size = 9;
N = window_size - 1;
k_ratio = 0.75;
R = k_ratio * N; % OS_CFAR的噪声功率选取索引
Pfa = 1e-3; % 虚警概率，经典值
K_factor = Pfa^(-1 / window_size) - 1; % 检测阈值因子
threshold_adjust = 60000;

detect_matrix_abs = abs(detect_matrix) .* abs(detect_matrix);% 平方律检测器
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
                % % CUT单元在检测窗口中的位置
                % if row_index(m)==i && col_index(n)==j
                %     CUT_pos_row = m;
                %     CUT_pos_col = n;
                % end
            end
        end
        % R维度逐行进行OS_CFAR
        % for row_index=1:size(detect_window,1)
        %     sorted_window = sort(detect_window, 2);
        % end
        sorted_window = sort(detect_window, 2);
        % D维度进行CA_CFAR
        noise_power = sum(sorted_window(:, R)) / length(sorted_window(:, R));
        % 检测门限
        threshold_matrix(i ,j) = noise_power * K_factor + threshold_adjust;
        if CUT>threshold_matrix(i ,j)
            target_index = [target_index ;i j];
        end
    end
    X = ['第', num2str(i), '行检测完毕'];
    disp(X);
end

end