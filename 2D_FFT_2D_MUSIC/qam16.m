function [complex_qam_data]=qam16(bitdata)
%modulation of 16QAM,modulate bitdata to 16QAM complex signal
X1=reshape(bitdata,4,length(bitdata)/4)';
d=1;%min distance of symble 
for i=1:length(bitdata)/4
    for j=1:4
        X1(i,j)=X1(i,j)*(2^(4-j));
    end
        source(i,1)=1+sum(X1(i,:));%convert to the number 1 to 16
end
mapping=[-3*d 3*d;
	   -d  3*d;
        d  3*d;
	  3*d  3*d;
	 -3*d  d;
	   -d  d;
	    d  d;
	  3*d  d;
 	 -3*d  -d; 
	   -d  -d; 
	    d  -d;
      3*d  -d;
	 -3*d  -3*d;
	   -d  -3*d;
	    d  -3*d;
	  3*d  -3*d];
 for i=1:length(bitdata)/4
     qam_data(i,:)=mapping(source(i),:);%data mapping
 end
 complex_qam_data=complex(qam_data(:,1),qam_data(:,2));
 