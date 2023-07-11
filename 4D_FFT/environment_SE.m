function [outputArg1] = environment_SE()
%ENVIRONMENT 生成成像环境
%针对车、行人、路灯、隔离带进行散射点建模
%% 移动汽车建模
tar_car=[2,0,0;2,0,0.9;2,0.6,0.9;2,1,1.4;2,4.5,1.4;2,5,0.9;2,6,0;
           0,0,0;0,0,0.9;0,0.6,0.9;0,1,1.4;0,4.5,1.4;0,5,0.9;0,6,0;
           2,0,0.5;2,1.5,1.7;2,2,1.8;2,2.5,1.8;2,3,1.8;2,3.5,1.8;2,4,1.7;2,5.5,0.8;2,6,0.5;
           0,0,0.5;0,1.5,1.7;0,2,1.8;0,2.5,1.8;0,3,1.8;0,3.5,1.8;0,4,1.7;0,5.5,0.8;0,6,0.5;
           2,3,1.4;2,3,0.6;2,3,0.3;2,3,0;2,1,0.9;2,1.4,0.9;2,1.8,0.9;2,2.2,0.9;2,2.6,0.9;2,3,0.9;2,3.4,0.9;2,3.8,0.9;2,4.2,0.9;2,4.6,0.9;2,1,0.5;2,5,0.4;2,0.6,0.4;2,0.4,0;2,1.4,0.4;2,1.6,0;2,5.4,0.3;2,5.6,0;2,4.6,0.3;2,4.4,0;2,2.1,0;2,2.6,0;2,3.5,0;2,4,0;%细节填充
           0,3,1.4;0,3,0.6;0,3,0.3;0,3,0;0,1,0.9;0,1.4,0.9;0,1.8,0.9;0,2.2,0.9;0,2.6,0.9;0,3,0.9;0,3.4,0.9;0,3.8,0.9;0,4.2,0.9;0,4.6,0.9;0,1,0.5;0,5,0.4;0,0.6,0.4;0,0.4,0;0,1.4,0.4;0,1.6,0;0,5.4,0.3;0,5.6,0;0,4.6,0.3;0,4.4,0;0,2.1,0;0,2.6,0;0,3.5,0;0,4,0;%细节填充
           1,0,0;1,0,0.9;1,0.6,0.9;1,5,0.9;1,6,0;1,4,1.7;1,1.5,1.7;1.5,3.2,1.8;0.5,3.2,1.8;1.5,2.5,1.8;0.5,2.5,1.8;1,6,0.5;
           2,1,-0.4;2,0.7,-0.3;2,1.3,-0.3;2,5,-0.4;2,4.7,-0.3;2,5.3,-0.3;%车轮
           0,1,-0.4;0,0.7,-0.3;0,1.3,-0.3;0,5,-0.4;0,4.7,-0.3;0,5.3,-0.3;%车轮
           2.2,4.6,0.9;-0.2,4.6,0.9];%后视镜
tar_car(:,3)=tar_car(:,3)+0.4;
tar_car(:,1)=tar_car(:,1)+0.2;

pos_car_1=[8,12];
car_group_1=[];
for i=1:size(pos_car_1,1)
    car_mid=tar_car;
    car_mid(:,1)=car_mid(:,1)+pos_car_1(i,1);
    car_mid(:,2)=car_mid(:,2)+pos_car_1(i,2);
    car_group_1=[car_group_1;car_mid];
end
pos_car_2=[18,2];
car_group_2=[];
for i=1:size(pos_car_2,1)
    car_mid=tar_car;
    car_mid(:,1)=car_mid(:,1)+pos_car_2(i,1);
    car_mid(:,2)=car_mid(:,2)+pos_car_2(i,2);
    car_group_2=[car_group_2;car_mid];
end
% 添加速度信息
v_car_1=ones(size(car_group_1,1),1)*10;
v_car_2=ones(size(car_group_2,1),1)*(-10);
car_group_1=[car_group_1 v_car_1];
car_group_2=[car_group_2 v_car_2];

car_group=[car_group_1;car_group_2];
% %% 隔离带建模
% plant_1=(0:0.5:20)'*[0,1,0];
% plant_1(:,1)=plant_1(:,1)+2;
% plant_1(:,3)=plant_1(:,3)+0;
% 
% plant_2=(0:0.5:20)'*[0,1,0];
% plant_2(:,1)=plant_2(:,1)+1;
% plant_2(:,3)=plant_2(:,3)+0;
% 
% plant_3=(0:0.5:20)'*[0,1,0];
% plant_3(:,1)=plant_3(:,1)+1;
% plant_3(:,3)=plant_3(:,3)+1;
% 
% plant_4=(0:0.5:20)'*[0,1,0];
% plant_4(:,1)=plant_4(:,1)+2;
% plant_4(:,3)=plant_4(:,3)+1;
% 
% plant_group1=[plant_1;plant_2;plant_3;plant_4];
% plant_group1(:,1)=plant_group1(:,1)+4;
% plant_group2=plant_group1;
% plant_group2(:,1)=plant_group1(:,1)+17;
% plant_group3=[plant_2;plant_3];
% plant_group3(:,1)=plant_group3(:,1)+13;
% % 添加速度信息
% v_plant_1=zeros(size(plant_group1,1),1);
% v_plant_2=zeros(size(plant_group2,1),1);
% v_plant_3=zeros(size(plant_group3,1),1);
% plant_group1=[plant_group1 v_plant_1];
% plant_group2=[plant_group2 v_plant_2];
% plant_group3=[plant_group3 v_plant_3];
% 
% plant_group=[plant_group1;plant_group2;plant_group3];
% %% 路灯建模
% light_1=[22,3,2;22,3,3;22,3,4;22,3,5;22,3,6;22,3,7;22,3,8;22,3,9;22,3,10;21,3,10;23,3,10;22,2,10;22,4,10];
% % light_2=light_1;
% % light_2(:,2)=light_2(:,2)+7;
% light_3=light_1;
% light_3(:,2)=light_3(:,2)+14;
% light_4=light_1;
% light_4(:,1)=light_4(:,1)-16;
% % light_5=light_2;
% % light_5(:,1)=light_5(:,1)-16;
% light_6=light_3;
% light_6(:,1)=light_6(:,1)-16;
% 
% % 添加速度信息
% v_light_1=zeros(size(light_1,1),1);
% % v_light_2=zeros(size(light_2,1),1);
% v_light_3=zeros(size(light_3,1),1);
% v_light_4=zeros(size(light_4,1),1);
% % v_light_5=zeros(size(light_5,1),1);
% v_light_6=zeros(size(light_6,1),1);
% light_1=[light_1 v_light_1];
% % light_2=[light_2 v_light_2];
% light_3=[light_3 v_light_3];
% light_4=[light_4 v_light_4];
% % light_5=[light_5 v_light_5];
% light_6=[light_6 v_light_6];
% 
% light_group=[light_1;light_3;light_4;light_6];
%% 行人建模
people=[0,0,0;0.5,0.5,0;0.25,0.3,1;0.25,0.3,1.5;0.25,0.3,1.7;0.4,0.4,0.5;0.1,0.2,0.5;0.25,0.3,1.8;0.5,0.3,1.4;0,0.3,1.4;0.5,0.5,1;0,0,0.9];
pos_p_1=[1,2;2,13];
people_group_1=[];
for i=1:size(pos_p_1,1)
    people_mid=people;
    people_mid(:,1)=people_mid(:,1)+pos_p_1(i,1);
    people_mid(:,2)=people_mid(:,2)+pos_p_1(i,2);
    people_group_1=[people_group_1;people_mid];
end
pos_p_2=[27,7;27,18];
people_group_2=[];
for i=1:size(pos_p_2,1)
    people_mid=people;
    people_mid(:,1)=people_mid(:,1)+pos_p_2(i,1);
    people_mid(:,2)=people_mid(:,2)+pos_p_2(i,2);
    people_group_2=[people_group_2;people_mid];
end
% 添加速度信息
v_people_1=ones(size(people_group_1,1),1)*2;
v_people_2=ones(size(people_group_2,1),1)*(-2);

people_group_1=[people_group_1 v_people_1];
people_group_2=[people_group_2 v_people_2];

people_group=[people_group_1;people_group_2];
%% 返回散射点集
% outputArg1 = car_group; %只有车
outputArg1 = [car_group;people_group]; % 车+人
% outputArg1 = [car_group;plant_group;light_group;people_group]; % 全部元素
end

