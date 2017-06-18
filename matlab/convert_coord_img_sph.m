function p = convert_coord_img_sph(pts, img_size, f)
% INPUT
%   p0:         m-by-2, the points
%   img_size:   2-by-1
%   f:          scalar

p0 = bsxfun(@minus, pts, wrev(img_size) / 2) / (max(img_size) / 2);
p = p0 * 18;
p = [p, f * ones(size(p0, 1), 1)];
lambda = atan(p(:,1) ./ p(:,3));
% lambda(lambda < 0) = lambda(lambda < 0) + pi;
phi = asin(p(:,2) ./ sqrt(sum(p.^2, 2)));
p = [lambda, phi];
end