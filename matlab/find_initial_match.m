function IDX = find_initial_match(feature1, feature2)
% This function finds the initial match.
%
% INPUT
%   feature1, feature2:     struct. including fields: 'pts',
%                           'polar_feature', 'sph'
% OUTPUT
%   IDX:    m-by-2, index array

dist_mat = pdist2(feature1.polar_feature, feature2.polar_feature, 'cosine');

% For a given point p1 in image1, find the most similar point p12 in image2,
% then find the point p21 in image1 that most similar to p12, check the
% distance between p1 and p21.
[D12, I12] = sort(dist_mat, 2);
[D21, I21] = sort(dist_mat);
D21 = D21'; I21 = I21';

ind = I21(I12(:,1),1) == (1:size(dist_mat,1))';
d_th = min(prctile(D12(:,1), 30), prctile(D21(:,1), 30));
ind = ind & D12(:,1) < d_th;
IDX = [find(ind), I12(ind, 1)];

xyz1 = [cos(feature1.sph(:,2)).*cos(feature1.sph(:,1)), cos(feature1.sph(:,2)).*sin(feature1.sph(:,1)), sin(feature1.sph(:,2))];
xyz2 = [cos(feature2.sph(:,2)).*cos(feature2.sph(:,1)), cos(feature2.sph(:,2)).*sin(feature2.sph(:,1)), sin(feature2.sph(:,2))];
theta = acos(sum(xyz1(IDX(:,1), :) .* xyz2(IDX(:,2), :), 2));
theta_th = min(prctile(theta, 75), pi/6);
IDX = IDX(theta < theta_th, :);
end
