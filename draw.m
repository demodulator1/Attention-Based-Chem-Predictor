file_name=["Attention","RBF","BP"];
file_name_tail="_loss_history.txt";
choose=3;

%% 画图

% 读取数据
data = readmatrix(strcat("history/",file_name(choose),file_name_tail));

% 提取训练损失和验证损失
train_loss = data(:, 1);
val_loss = data(:, 2);
% MSE = data(:, 3);

% 绘制损失曲线
figure;
plot(train_loss, ':k', 'LineWidth', 2, 'DisplayName', 'Training loss'); % 黑色点线
hold on;
plot(val_loss, '-k', 'LineWidth', 2, 'DisplayName', 'Validation loss'); % 黑色实线
% plot(val_loss, '--k', 'LineWidth', 2.5, 'DisplayName', 'Train MSE'); % 黑色虚线

% 设置轴标签和标题，使用LaTeX解释器
xlabel('Epoch', 'FontSize', 12, 'Interpreter', 'latex');
ylabel('Loss', 'FontSize', 12, 'Interpreter', 'latex');
title('Training and Validation Loss', 'FontSize', 14, 'Interpreter', 'latex');

% 设置图例
legend('show', 'Location', 'northeast', 'FontSize', 12, 'Interpreter', 'latex');

% 设置图形属性
set(gca, 'FontSize', 12, 'LineWidth', 1);
grid on;
box on;

% 设置背景为白色
set(gcf, 'Color', 'w');

%% 导出



% 导出图片
exportgraphics(gcf, strcat('output/', file_name(choose), '_loss_plot.png'), 'Resolution', 300);

