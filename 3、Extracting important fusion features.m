f = find(weight <= 0.5);   
fc1 = fc;
fc1(f, :) = []; 
weight1 = weight;
weight1(f) = [];          
len = zeros(1, 4050);
for i = 1:4050
    [u, v] = find(fc1 == i);
    len(i) = sum(weight1(u));
end
[Freq, Freq_pos] = sort(len, 'descend');
Important_fc = Freq_pos(1:400);
FC_Important = FC(:, Important_fc);
