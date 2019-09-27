
load('E:\��Ȩ���ɭ��ʵ��\1_105�����ԵĶ�ά������.mat')

clearvars -except FC

% 1����ʼ��������ǰ55������ΪPD����-1����ʾ����50������ΪNC����+1�����б�ʾ
PD = -ones(1, 55);
NC = ones(1, 50);
labels = [PD, NC];

% 2�����ݼ��Ļ���
n = 65;     % ���ֵĸ����ֱ��ǣ�35:30:40����ǰ65����Ϊѵ����+��֤����
            % ���40����Ϊ���Լ�

% �������б��Եı��
x = randperm(105);
y = x(1:n);             % y��Ϊѵ����+��֤����  �±�
z = x(n+1:105);         % z��Ϊ���Լ������ݵ�   �±�
 
for N = 20:10:600           % N�Ǽ�Ⱥ�л�����������Ŀ
    test_num = length(z);   % test_numΪ���Լ���������Ŀ
    test_lab = labels(z);   % test_labΪ���Լ���ʵ�ʱ�ǩ
    
    fc_num = 64;           % fc_numΪÿ�γ�ȡ��������
    sam_num = 35;          % ��ѵ����֤��y���޷Żس�ȡ����������Ŀ����Ϊѵ����
    
    weight = zeros(1, N);   % ��ΪN������������Ȩ��
    fc = zeros(N, fc_num);  % fcΪÿ������������ȡ���������
    
    val_out = zeros(N, 30);
    test_out = zeros(N, test_num);
    
    for i = 1:N
        
        flag = zeros(1, n);
     % ��y�������ȡ35����Ϊѵ��������ֵ������train
        sample_num = randsample(length(y), sam_num, 'false');
        flag(sample_num) = 1;
        train = y(sample_num);      % train��Ϊ��ȡ����ѵ��������
        train_lab = labels(train);
        
        fc(i, :) = randsample(4050, fc_num, 'false');
        fc_temp = fc(i, :);
        
     % ������֤����y��ʣ���30����Ϊ��֤��
        validate = y(flag == 0);
        val_lab = labels(validate);
        
  % 3��ʹ��ѵ����ѵ��ģ��
        train_set = FC(train, fc_temp);
        tree = fitctree(train_set, train_lab);
  % 4��ʹ��ģ�Ͷ���֤�����з��࣬��ÿ�����ķ���׼ȷ����Ϊ����������Ȩ��
        validate_set = FC(validate, fc_temp);
        val_out(i, :) = predict(tree, validate_set);
        weight(i) = sum(val_lab == val_out(i, :))/length(val_lab);
        
  % 5��ʹ�����ɭ�ֶԲ��Լ����������з���
        test_set = FC(z, fc_temp);
        test_out(i, :) = predict(tree, test_set);
        
    end
  % 6����ȨͶƱ����Ϊģ�͵����շ�����
    result = zeros(1, length(z));
    for i = 1:length(z)
        result(i) = sum(test_out(:, i) .* weight');
    end
    result1 = sign(result);
    R = sum(result1 == test_lab);
    E(N) = R/length(z);
end

% 7������ͼ��
E(E == 0) = [];
X = [20:10:600];
figure;
plot(X, E);
xlabel('����������Ŀ', 'fontname', '����', 'fontsize', 16)
ylabel('��Ȩ���ɭ��׼ȷ��', 'fontname', '����', 'fontsize', 16)
axis([0 600 0 1])
