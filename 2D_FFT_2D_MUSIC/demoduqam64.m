function [demodu_bit_symble]=demoduqam64(Rx_serial_complex_symbols)
% 8QAM���
complex_symbols=reshape(Rx_serial_complex_symbols,length(Rx_serial_complex_symbols),1);
d=1;
mapping=[-7*d 7*d;
        -5*d 7*d;
	     -3*d 7*d;
        -1*d 7*d;
        1*d 7*d;
        3*d 7*d;
        5*d 7*d;
        7*d 7*d;
        -7*d 5*d;
        -5*d 5*d;
	     -3*d 5*d;
        -1*d 5*d;
        1*d 5*d;
        3*d 5*d;
        5*d 5*d;
        7*d 5*d;
        -7*d 3*d;
        -5*d 3*d;
	     -3*d 3*d;
        -1*d 3*d;
        1*d 3*d;
        3*d 3*d;
        5*d 3*d;
        7*d 3*d;
        -7*d 1*d;
        -5*d 1*d;
	     -3*d 1*d;
        -1*d 1*d;
        1*d 1*d;
        3*d 1*d;
        5*d 1*d;
        7*d 1*d;
        -7*d -1*d;
        -5*d -1*d;
	     -3*d -1*d;
        -1*d -1*d;
        1*d -1*d;
        3*d -1*d;
        5*d -1*d;
        7*d -1*d;
        -7*d -3*d;
        -5*d -3*d;
	     -3*d -3*d;
        -1*d -3*d;
        1*d -3*d;
        3*d -3*d;
        5*d -3*d;
        7*d -3*d;
        -7*d -5*d;
        -5*d -5*d;
	     -3*d -5*d;
        -1*d -5*d;
        1*d -5*d;
        3*d -5*d;
        5*d -5*d;
        7*d -5*d;
        -7*d -7*d;
        -5*d -7*d;
	     -3*d -7*d;
        -1*d -7*d;
        1*d -7*d;
        3*d -7*d;
        5*d -7*d;
        7*d -7*d];
  complex_mapping=complex(mapping(:,1),mapping(:,2));
  for i=1:length(Rx_serial_complex_symbols)
      for j=1:64
          metrics(j)=abs(complex_symbols(i,1)-complex_mapping(j,1));
      end
      [min_metric  decode_symble(i)]= min(metrics) ;  %将离某星座点�?近的值赋给decode_symble(i)
  end
  
  decode_bit_symble=de2bi((decode_symble-1)','left-msb');
  demodu_bit_symble=reshape(decode_bit_symble',1,length(Rx_serial_complex_symbols)*6);
      