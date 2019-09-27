% 03_ȷ����������
clearvars -except FC Important_fc FC_Important
% 1����ʼ��������ǰ55������ΪPD����-1����ʾ����50������ΪNC����+1�����б�ʾ
PD = -ones(1, 55);
NC = ones(1, 50);
labels = [PD, NC];
 
n = 65;     % ���ֵĸ����ֱ��ǣ�35:30:40����ǰ75����Ϊѵ����+��֤����
            % ���40����Ϊ���Լ�
% 2�������ȡ�����Ĺ���������ʾ��
  % �������б��Եı��
x = randperm(105);
y = x(1:n);             % x��Ϊѵ����+��֤����  �±�
z = x(n+1:105);         % z��Ϊ���Լ������ݵ�   �±�

N = 410;                % N�Ǽ�Ⱥ�л�����������Ŀ
test_num = length(z);   % test_numΪ���Լ���������Ŀ
test_lab = labels(z);
 
fc_num = 64;            % ÿ�γ�ȡ��������
sam_num = 35;           % ��ѵ����֤�����޷Żس�ȡ����������Ŀ
 
weight = zeros(1, N);   % ��ΪN������������Ȩ��
fc = zeros(N, fc_num);  % fcΪÿ������������ȡ���������
 
val_out = zeros(N, 30);
test_out = zeros(N, test_num);
 
for k = 70:2:400
    for i = 1:N
        
        flag = zeros(1, n);
      % ��y�������ȡ35����Ϊѵ��������ֵ������train
        sample_num = randsample(length(y), sam_num, 'false');
        flag(sample_num) = 1;
        train = y(sample_num);      % train��Ϊ��ȡ����ѵ��������
        train_lab = labels(train);
        
        fc(i, :) = randsample(k, fc_num, 'false');
        fc_temp = fc(i, :);
        
      % ������֤����y��ʣ���30����Ϊ��֤��
        validate = y(flag == 0);
        val_lab = labels(validate);
        
  % 3��ʹ��ѵ����ѵ��ģ��
        train_set = FC_Important(train, fc_temp);
        tree = fitctree(train_set, train_lab);
        
  % 4��ʹ��ģ�Ͷ���֤�����з��࣬��ÿ�����ķ���׼ȷ����Ϊ����������Ȩ��
        validate_set = FC_Important (validate, fc_temp);
        val_out(i, :) = predict(tree, validate_set);
        weight(i) = sum(val_lab == val_out(i, :))/length(val_lab);
  % 5��ʹ�����ɭ�ֶԲ��Լ����������з���
        test_set = FC_Important (z, fc_temp);
        test_out(i, :) = predict(tree, test_set);
    end
  % 6����ȨͶƱ����Ϊģ�͵����շ�����
    result = zeros(1, length(z));
    for i = 1:length(z)
        result(i) = sum(test_out(:, i) .* weight');
    end
    result1 = sign(result+20);
    R = sum(result1 == test_lab);
    E(k) = R/length(z);
end
% 7������Ѱ������������Ŀ��ͼ��
E(E == 0) = [];
X = [70:2:400];
plot(X, E)
xlabel('��Ҫ������Ŀ', 'fontname', '����', 'fontsize', 16)
ylabel('��Ȩ���ɭ��׼ȷ��', 'fontname', '����', 'fontsize', 16)
title('ѡ������������Ŀͼ', 'fontname', '����', 'fontsize', 16)
axis([0 400 0 1])
