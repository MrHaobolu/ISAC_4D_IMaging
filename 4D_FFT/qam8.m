function [complex_qam_data]=qam8(bitdata)
%modulation of 16QAM,modulate bitdata to 16QAM complex signal
bit_per_symbol = 3;
X1=reshape(bitdata,bit_per_symbol,length(bitdata)/bit_per_symbol)';
d=1;%min distance of symble 
for i=1:length(bitdata)/bit_per_symbol
    for j=1:bit_per_symbol
        X1(i,j)=X1(i,j)*(2^(bit_per_symbol-j));
    end
        source(i,1)=1+sum(X1(i,:));%convert to the number 1 to 8
end
mapping=[-d d;
	   0  d;
        d  d;
	 -d  0;
	   d  0;
 	 -d  -d; 
	   0  -d; 
	    d  -d];
 for i=1:length(bitdata)/bit_per_symbol
     qam_data(i,:)=mapping(source(i),:);%data mapping
 end
 complex_qam_data=complex(qam_data(:,1),qam_data(:,2));
 