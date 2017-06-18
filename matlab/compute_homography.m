function [matH, pair_idx] = compute_homography(pts1, pts2, varargin)
% This function estimate Homography matrix. Homography matrix H is defined
% as the matrix corresponds points in two images.
%
%       x'_i = H * x_i
%
% where x_i and x'_i are corrensponding points written in homogeneous
% coordinates. x_i = [u_i, v_i, 1].
% The homography matrix are determined up to an arbitrary scale factor.
%
%           h11 * u_i + h12 * v_i + h13
%   u'_i = -----------------------------
%           h31 * u_i + h32 * v_i + h33
%
%           h21 * u_i + h22 * v_i + h23
%   v'_i = -----------------------------
%           h31 * u_i + h32 * v_i + h33
% the two equations can be written in matrix form as:
%   kron([u, v, 1], [1, 0, -u'; 0, 1, -v']) * h = 0
% where h = vec(H), or, H = reshape(h, 3, 3)
%
% INPUT
%   pts1, pts2:     n-by-2 matrix, unit in pixels
% OUTPUT
%   matH:           3-by-3 matrix
%   pair_idx:       m-by-1
% PROPERTIES
%   InitialThreshold:   double
%   MaxTrialNumber:     double
%   FineTune:           true/false

% Assertions
assert(all(size(pts1) == size(pts2)) && size(pts1, 2) == 2);
assert(size(pts1, 1) > 15);

pts1 = double(pts1);
pts2 = double(pts2);

% Deal with some parameters
initial_threshold = 5;
max_trial_number = 1000;
need_fine_tune = true;
if ~isempty(varargin)
    if mod(length(varargin), 2) ~= 0
        error('Wrong number of arguments!');
    end
    for i = 1:length(varargin)/2
        key = varargin{i*2-1};
        value = varargin{i*2};
        switch lower(key)
            case 'initialthreshold'
                initial_threshold = value;
            case 'maxtrialnumber'
                max_trial_number = value;
            case 'finetune'
                if ~islogical(value)
                    error('Wrong type of property value!');
                end
                need_fine_tune = value;
            otherwise
                error('Wrong property name!');
        end
    end
end
pair_num = size(pts1, 1);

pts_mean = mean([pts1; pts2]);
pts_std = std([pts1; pts2]);
matT = [diag(sqrt(2)./pts_std), -pts_mean(:)*sqrt(2)./pts_std(:); 0, 0, 1];

pts1_normalized = bsxfun(@times, bsxfun(@minus, pts1, pts_mean), sqrt(2)./pts_std);
pts2_normalized = bsxfun(@times, bsxfun(@minus, pts2, pts_mean), sqrt(2)./pts_std);

% RANSAC
inliers = 0;
th = initial_threshold;
ransac_num = 6;     % At least 4 points. Here use 6 points for robustness.
k = 0;
% while inliers < max(pair_num * 0.5, 20)
while inliers < max(pair_num * 0.5, 10)
    idx = datasample(1:pair_num, ransac_num, 'Replace', false);
    
    uv1 = pts1_normalized(idx, :);
    uv2 = pts2_normalized(idx, :);
    matH = solve_dlt(uv1, uv2);
    [err1, err2] = compute_reproj_err(pts1, pts2, matT\matH*matT);
    inliers = sum(err1 < th & err2 < th);
    
    k = k + 1;
    if k > max_trial_number
        warning('MaxTrailNumber reached.');
        break;
    end
end

d = inf;
k = 0;
while abs(d) > 0
    idx = err1 < th & err2 < th;
    uv1 = pts1_normalized(idx, :);
    uv2 = pts2_normalized(idx, :);
    
    matH = solve_dlt(uv1, uv2);
    [err1, err2] = compute_reproj_err(pts1, pts2, matT\matH*matT);
    
    th = median([err1; err2]) * 2.5;
    d = sum(idx) - sum(err1 < th & err2 < th);
    if d < 0
        k = k + 1;
    end
    if k > 5
        warning('aaa!!!');
        break;
    end
end

matH = matT \ matH * matT;
matH = matH / norm(matH(:));
pair_idx = err1 < th & err2 < th;
pair_idx = find(pair_idx);

if need_fine_tune
    [matH, idx] = fine_tune_matH(pts1(pair_idx, :), pts2(pair_idx, :), matH);
    pair_idx = pair_idx(idx);
end
end


function matH = solve_dlt(uv1, uv2)
% Helper function
% This function helps to compute the homography matrix
%       uv2 = [uv1, 1] * matH'

assert(all(size(uv1) == size(uv2)));
assert(size(uv1, 2) == 2);

n = size(uv1, 1);
A = nan(n * 2, 9);
for i = 1:n
    A(i*2-1:i*2, :) = kron([uv1(i, :), 1], [eye(2), -uv2(i, :)']);
end
[~, ~, V] = svds(A, 9);
matH = reshape(V(:, end), 3, 3);

end


function [err1, err2] = compute_reproj_err(uv1, uv2, matH)
% Helper function
% This function helps to compute the reprojection error

assert(all(size(uv1) == size(uv2)));
assert(size(uv1, 2) == 2);

n = size(uv1, 1);
tmp_pts = [uv1, ones(n, 1)] * matH';
tmp_pts = bsxfun(@times, tmp_pts(:, 1:2), 1./tmp_pts(:,3));
err2 = sqrt(sum((uv2 - tmp_pts).^2, 2));

tmp_pts = [uv2, ones(n, 1)] / matH';
tmp_pts = bsxfun(@times, tmp_pts(:, 1:2), 1./tmp_pts(:,3));
err1 = sqrt(sum((uv1 - tmp_pts).^2, 2));
end


function [matH, idx] = fine_tune_matH(pts1, pts2, matH)
% Helper function

assert(all(size(pts1) == size(pts2)));
assert(size(pts1, 2) == 2);
n = size(pts1, 1);

d = inf; k = 0;
idx = true(n, 1);
while abs(d) > 0
    uv1 = pts1(idx, :);
    uv2 = pts2(idx, :);
    
    h = fminsearch(@(h)finetune_helper(uv1, uv2, reshape(h, 3, 3)), matH(:));
    matH = reshape(h / norm(h), 3, 3);
    [err1, err2] = compute_reproj_err(pts1, pts2, matH);
    
    th = median([err1; err2]) * 2.5;
    d = sum(idx) - sum(err1 < th & err2 < th);
    idx = err1 < th & err2 < th;
    if d < 0
        k = k + 1;
    end
    if k > 5
        warning('aaa!!!');
        break;
    end
end
end


function e = finetune_helper(uv1, uv2, matH)
[err1, err2] = compute_reproj_err(uv1, uv2, matH);
e = sum(err1 + err2);
end
