function [chosen_index]=optimal_design_space(Theta,chosen_index,n_s,X)
[m_x,~] = size(Theta);
for i=1:n_s
    space_filling = zeros(m_x,1);
    
    for j=1:m_x
        if ~ismember(j,chosen_index)
            space = X(j,:)-X(chosen_index,:);
            space_filling(j,1) = min(sqrt(sum(space.^2,2)));
            
        end
    end
    space_filling = space_filling./max(space_filling);
    temp = space_filling;
    optimal = max(temp);
    index = find(temp==optimal,1,'first');
    chosen_index = [chosen_index;index];
end

end