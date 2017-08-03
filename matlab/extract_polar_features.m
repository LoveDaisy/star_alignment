function [rho, theta, vol_res] = extract_polar_features(sph, vol, K)
% This function extract the polar features of a point
m = K*2;
% vec = [cos(sph(:,1)) .* cos(sph(:,2)), cos(sph(:,1)) .* sin(sph(:,2)), sin(sph(:,1))];
vec = [cos(sph(:,2)) .* cos(sph(:,1)), cos(sph(:,2)) .* sin(sph(:,1)), sin(sph(:,2))];

[dist_mat, ind] = pdist2(vec, vec, 'cosine', 'smallest', m);
dist_mat = acos(1 - dist_mat);
vol = vol(ind(2:end, :));
[~, vol_ind] = sort(vol.*dist_mat(2:end,:), 'descend');
% vol_ind = repmat((1:m)', 1, size(ind,2));

angular_features = zeros(size(sph, 1), K);
for i = 1:size(sph, 1)
    v0 = vec(i, :);
    vs = vec(ind(vol_ind(1:K, i)+1, i), :);
    angles = vs * make_cross_mat(v0)';
%     angles = zeros(size(vs, 1), 3);
%     for j = 1:size(vs, 1)
%         angles(j, :) = cross(v0, vs(j, :));
%     end
    angles = bsxfun(@times, angles, 1./sqrt(sum(angles.^2, 2)));
    cr = angles * make_cross_mat(angles(1,:))';
    s = sqrt(sum(cr.^2, 2)) .* sign(cr * v0');
    c = angles * angles(1, :)';
    a = asin(s);
    a(c < 0 & s > 0) = pi - a(c < 0 & s > 0);
    a(c < 0 & s < 0) = -pi - a(c < 0 & s < 0);
    angular_features(i, :) = a;
%     for j = 1:size(vs, 1)
%         cr = cross(angles(1, :), angles(j, :));
%         s = norm(cr) * sign(cr * v0');
%         c = angles(1, :) * angles(j, :)';
%         a = asin(s);
%         if c < 0 && s > 0
%             a = pi - a;
%         elseif c < 0 && s < 0
%             a = -pi - a;
%         end
%         angular_features(i, j) = a;
%     end
end

rho = zeros(size(sph, 1), K);
vol_res = zeros(size(sph, 1), K);
for i = 1:size(sph, 1)
    rho(i, :) = dist_mat(vol_ind(1:K, i)+1, i)';
    vol_res(i, :) = vol(vol_ind(1:K, i), i)';
end
theta = angular_features;
end


function crossMat = make_cross_mat(v)
% This function makes a cross matrix
% cross(v, a) = crossMat * a
crossMat = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end
