%%  清空环境变量
warning off
close all
clear
clc

%%  导入数据
res = xlsread('clinical.xlsx');

%%  划分训练集和测试集
temp = randperm(200);

P_train = res(temp(1:140), 1:3)';
T_train = res(temp(1:140), 4)';
M = size(P_train, 2);

P_test  = res(temp(141:end), 1:3)';
T_test  = res(temp(141:end), 4)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

%%  转置以适应模型
p_train = p_train'; 
p_test  = p_test';
t_train = T_train'; 
t_test  = T_test';

%%  训练逻辑回归模型——多分类逻辑回归模型（ECOC）
t = templateLinear( ...
    'Learner', 'logistic', ...
    'Regularization', 'ridge');

net = fitcecoc(p_train, t_train, ...
    'Learners', t, ...
    'Coding', 'onevsall');

%%  仿真测试
T_sim1 = predict(net, p_train);
T_sim2 = predict(net, p_test);

%%  性能评价
error1 = sum(T_sim1' == T_train) / M * 100;
error2 = sum(T_sim2' == T_test ) / N * 100;

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  绘图
figure
plot(1:M, T_train, 'r-*', 1:M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
title({'训练集预测结果对比'; ['准确率=' num2str(error1) '%']})
grid

figure
plot(1:N, T_test, 'r-*', 1:N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
title({'测试集预测结果对比'; ['准确率=' num2str(error2) '%']})
grid

%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%%  混淆矩阵（合并）
a1 = [T_train, T_test];
b1 = [T_sim1', T_sim2'];
a1 = full(ind2vec(a1));
b1 = full(ind2vec(b1));

figure
plotconfusion(a1, b1);
