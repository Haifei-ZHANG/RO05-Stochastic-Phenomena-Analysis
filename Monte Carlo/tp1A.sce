T=1;
h=0.05;
M=T/h;
S0=1;
miu=0.05;
sigma=0.3;
Sn=zeros(1000,M+1);
Sn(:,1)=S0;
for i=1:1000
    u=rand(20,1);
    v=rand(20,1);
    for j=2:M+1
      ksi_n=sqrt(log(u(j-1))*-2)*cos(2*%pi*v(j-1)); 
      Sn(i,j)=Sn(i,j-1)*(1+miu*h+sigma*sqrt(h)*ksi_n);
    end  
end

moyenne_SM=mean(Sn(:,21));
x=linspace(0,1,M+1);
plot(x,Sn(:,:));
