% ���Ĳ��֣��ҵ����������Ļ��������
% 1���������ŵ��������ϣ�ǰ320ά������������
Important_fc = Important_fc(1:320);
FC_Important = FC_Important(:, 1:320);
% 2������һ��������Ϊ�����š���Ϊ�������
t = 1; 
for i = 1:45
    for j = 1:90
        a(i, j) = t;
        t = t + 1;
    end
end
% 3��XX�ǻ����ţ�YY��������š���˴�ʱ�͵õ���Ӧ�Ľ�����
for i = 1:320
    [XX(i), YY(i)] = find(a == Important_fc(i));
end
% 4��Ȼ����������������Ƶ��
% ��1 ����������Ƶ��
brain =zeros(1,90);
for i = 1:90
    brain (i) = length(find(YY == i));
end
% rank_bΪbrain�Ľ�������brain_pos_bΪ��ԭʼ������±ꡣ
[rank_b,brain_pos_b]=sort(brain,'descend'); 
brain_new=[rank_b; brain_pos_b];   %��������±���ϳ�һ���¾���
% ��2 ��������Ƶ��
jiyin = zeros(1, 45);
for i = 1:45
    jiyin(i) = length(find(XX == i));
end
% rank_jΪjiyin�Ľ�������brain_pos_jΪ��ԭʼ������±ꡣ
[rank_j,brain_pos_j]=sort(jiyin,'descend');
jiyin_new=[rank_j; brain_pos_j];  % ��������±���ϳ�һ���¾���
