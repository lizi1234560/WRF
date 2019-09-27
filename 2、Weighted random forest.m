
load('E:\加权随机森林实验\1_105个被试的二维特征表.mat')

clearvars -except FC

% 1）初始化变量，前55个被试为PD，用-1来表示，后50个被试为NC，用+1来进行表示
PD = -ones(1, 55);
NC = ones(1, 50);
labels = [PD, NC];

% 2）数据集的划分
n = 65;     % 划分的个数分别是：35:30:40；将前65个作为训练集+验证集；
            % 最后40个作为测试集

% 打乱所有被试的编号
x = randperm(105);
y = x(1:n);             % y作为训练集+验证集的  下标
z = x(n+1:105);         % z作为测试集的数据的   下标
 
for N = 20:10:600           % N是集群中基分类器的数目
    test_num = length(z);   % test_num为测试集的样本数目
    test_lab = labels(z);   % test_lab为测试集的实际标签
    
    fc_num = 64;           % fc_num为每次抽取的特征数
    sam_num = 35;          % 从训练验证集y中无放回抽取的样本的数目，作为训练集
    
    weight = zeros(1, N);   % 作为N个基分类器的权重
    fc = zeros(N, fc_num);  % fc为每个基分类器抽取的特征编号
    
    val_out = zeros(N, 30);
    test_out = zeros(N, test_num);
    
    for i = 1:N
        
        flag = zeros(1, n);
     % 从y中随机抽取35个作为训练集，赋值给变量train
        sample_num = randsample(length(y), sam_num, 'false');
        flag(sample_num) = 1;
        train = y(sample_num);      % train作为抽取到的训练集样本
        train_lab = labels(train);
        
        fc(i, :) = randsample(4050, fc_num, 'false');
        fc_temp = fc(i, :);
        
     % 划分验证集，y中剩余的30个作为验证集
        validate = y(flag == 0);
        val_lab = labels(validate);
        
  % 3）使用训练集训练模型
        train_set = FC(train, fc_temp);
        tree = fitctree(train_set, train_lab);
  % 4）使用模型对验证集进行分类，将每棵树的分类准确率作为基分类器的权重
        validate_set = FC(validate, fc_temp);
        val_out(i, :) = predict(tree, validate_set);
        weight(i) = sum(val_lab == val_out(i, :))/length(val_lab);
        
  % 5）使用随机森林对测试集的样本进行分类
        test_set = FC(z, fc_temp);
        test_out(i, :) = predict(tree, test_set);
        
    end
  % 6）加权投票，作为模型的最终分类结果
    result = zeros(1, length(z));
    for i = 1:length(z)
        result(i) = sum(test_out(:, i) .* weight');
    end
    result1 = sign(result);
    R = sum(result1 == test_lab);
    E(N) = R/length(z);
end

% 7）画出图形
E(E == 0) = [];
X = [20:10:600];
figure;
plot(X, E);
xlabel('基分类器数目', 'fontname', '楷体', 'fontsize', 16)
ylabel('加权随机森林准确率', 'fontname', '楷体', 'fontsize', 16)
axis([0 600 0 1])
