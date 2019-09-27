

% 变量的初始化
humnum = size(genes_105, 1); % 获取被试的数目
gennum = size(genes_105, 2); % 获取基因的数目
% branum = size(times, 2);% 获取脑区的数目
z = zeros(80, 1);


% 初始化变量feature为（74，36×36）的矩阵
feature_snp = nan(humnum, 990);
 

% 计算每个被试的基因和脑区时间序列的相关系数
for k = 1:humnum                  % k表示的是人数， [1, 105]
    
    x = 1;
    for i = 2:gennum              % i表示基因的数目，[1, 45]
        for j = 1:i-1

          % 将变量转化成90行1列形式的变量
            gene = genes_105(k, i, :);
            gen = reshape(gene, 80, 1);
            
            time = genes_105(k, j, :);
            tim = reshape(time, 80, 1);
            
            
          % 如果基因数据或者时间序列全为0，则将相关系数赋值为0
%             if isequal(z, gen) || isequal(tim, z)
%                 feature(k, branum*(i-1) + j) = 0;
%                 continue;
%             end
            
            
            R = corrcoef(gen, tim);                   % 计算相关系数矩阵
%             R = corrcoef(U(:, 1), V(:, 1));           % 计算相关系数
            
            feature_snp(k, x) = R(1, 2);
            x = x+1;
        end
    end
end


