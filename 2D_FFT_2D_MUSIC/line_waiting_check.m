    % 构建阵列流形开始搜索
    for theta_index=1:length(theta_list)
        theta_search = theta_list(theta_index)*pi/180;
        for faii_index=1:length(faii_list)
            faii_search = faii_list(faii_index)*pi/180;
            % 线阵阵列流行置0
            W_search_azimuth = zeros(K_sub,1);
            W_search_pitch = zeros(K_sub,1);
            % 线阵1
            for index_y=1:K_sub
                index_x = 1;
                % 波程差为-，且距离关系发生变化
                if theta_search>(90*pi/180)
                    if faii_search<=(90*pi/180)
                        r=(index_x-1)*d*cos(pi-theta_search)-(index_y-1)*d*sin(pi-theta_search);%波程差水平投影
                        W_search_azimuth(index_y, 1)=exp(-1j*2*pi*r*cos(faii_search)/lambda); 
                    end
                    if faii_search>(90*pi/180)
                        r=(index_x-1)*d*cos(pi-theta_search)+(index_y-1)*d*sin(pi-theta_search);%波程差水平投影
                        W_search_azimuth(index_y, 1)=exp(-1j*2*pi*r*(cos(pi-faii_search))/lambda); 
                    end
                end
                %波程差为+
                if theta_search<=(90*pi/180)

                    if faii_search<=(90*pi/180)
                        r=(index_x-1)*d*cos(theta_search)+(index_y-1)*d*sin(theta_search);%波程差水平投影
                        W_search_azimuth(index_y, 1)=exp(1j*2*pi*r*cos(faii_search)/lambda);
                    end
                    if faii_search>(90*pi/180)
                        r=(index_x-1)*d*cos(theta_search)-(index_y-1)*d*sin(theta_search);%波程差水平投影
                        W_search_azimuth(index_y, 1)=exp(1j*2*pi*r*(cos(pi-faii_search))/lambda); 
                    end
                end
            end
            % 线阵2
            for index_x=1:K_sub
                index_y = 1;
                % 波程差为-，且距离关系发生变化
                if theta_search>(90*pi/180)
                    if faii_search<=(90*pi/180)
                        r=(index_x-1)*d*cos(pi-theta_search)-(index_y-1)*d*sin(pi-theta_search);%波程差水平投影
                        W_search_pitch(index_x, 1)=exp(-1j*2*pi*r*cos(faii_search)/lambda); 
                    end
                    if faii_search>(90*pi/180)
                        r=(index_x-1)*d*cos(pi-theta_search)+(index_y-1)*d*sin(pi-theta_search);%波程差水平投影
                        W_search_pitch(index_x, 1)=exp(-1j*2*pi*r*(cos(pi-faii_search))/lambda); 
                    end
                end
                %波程差为+
                if theta_search<=(90*pi/180)

                    if faii_search<=(90*pi/180)
                        r=(index_x-1)*d*cos(theta_search)+(index_y-1)*d*sin(theta_search);%波程差水平投影
                        W_search_pitch(index_x, 1)=exp(1j*2*pi*r*cos(faii_search)/lambda);
                    end
                    if faii_search>(90*pi/180)
                        r=(index_x-1)*d*cos(theta_search)-(index_y-1)*d*sin(theta_search);%波程差水平投影
                        W_search_pitch(index_x, 1)=exp(1j*2*pi*r*(cos(pi-faii_search))/lambda); 
                    end
                end
            end
    
            o_matrix_azimuth = (W_search_azimuth'*En_azimuth)*(En_azimuth'*W_search_azimuth);
            o_matrix_pitch = (W_search_pitch'*En_pitch)*(En_pitch'*W_search_pitch);
            
            o_matrix_azimuth_m(theta_index, faii_index) = abs(1./o_matrix_azimuth);
            o_matrix_pitch_m(theta_index, faii_index) = abs(1./o_matrix_pitch);
            
           
        end
    end