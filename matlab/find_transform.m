function tf = find_transform(pts1, pts2, pair_idx)
% This function find transform between images. The result tf
% transform image1 to image2.
%   img_tf = imwarp(img, tf, 'OutputView', ref);
%
% INPUT
%   pts1, pts2:     struct. including fields of:
%                   'pts', 'polar_feature'
% OUTPUT
%   tf:     tform, the output of projective2d() function

% Find the perspective transformation using the initial match
[matH, ~] = compute_homography(pts1(pair_idx(:,1),:), pts2(pair_idx(:,2),:));

% Use all points to fine tune the matH
p0 = [pts1, ones(size(pts1, 1), 1)] * matH';
p0 = bsxfun(@times, p0(:,1:2), 1./p0(:,3));
dist_mat = pdist2(p0, pts2);
[min_dist, ind] = min(dist_mat, [], 2);
pair_idx = [(1:size(pts1,1))', ind];
pair_idx = pair_idx(min_dist < 5, :);
[matH, ~] = compute_homography(pts1(pair_idx(:,1),:), pts2(pair_idx(:,2),:));
% pair_idx = pair_idx(pair_idx_,:);
    
% The result
% pts1 = pts1(pair_idx(:,1),:);
% pts2 = pts2(pair_idx(:,2),:);
tf = projective2d(matH');
end
