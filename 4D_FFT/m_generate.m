function[mg]=m_generate(f)
n=length(f);%��λ�Ĵ�������
N=2^n-1;%α����������
register=[zeros(1,n-1) 1];
for i=1:N
    newregister(1)=mod(sum(f.*register),2);
    for j=2:n
        newregister(j)=register(j-1);
    end
    register=newregister;
    mg(i)=register(n);
end
