

% �����ĳ�ʼ��
humnum = size(genes_105, 1); % ��ȡ���Ե���Ŀ
gennum = size(genes_105, 2); % ��ȡ�������Ŀ
% branum = size(times, 2);% ��ȡ��������Ŀ
z = zeros(80, 1);


% ��ʼ������featureΪ��74��36��36���ľ���
feature_snp = nan(humnum, 990);
 

% ����ÿ�����ԵĻ��������ʱ�����е����ϵ��
for k = 1:humnum                  % k��ʾ���������� [1, 105]
    
    x = 1;
    for i = 2:gennum              % i��ʾ�������Ŀ��[1, 45]
        for j = 1:i-1

          % ������ת����90��1����ʽ�ı���
            gene = genes_105(k, i, :);
            gen = reshape(gene, 80, 1);
            
            time = genes_105(k, j, :);
            tim = reshape(time, 80, 1);
            
            
          % ����������ݻ���ʱ������ȫΪ0�������ϵ����ֵΪ0
%             if isequal(z, gen) || isequal(tim, z)
%                 feature(k, branum*(i-1) + j) = 0;
%                 continue;
%             end
            
            
            R = corrcoef(gen, tim);                   % �������ϵ������
%             R = corrcoef(U(:, 1), V(:, 1));           % �������ϵ��
            
            feature_snp(k, x) = R(1, 2);
            x = x+1;
        end
    end
end


