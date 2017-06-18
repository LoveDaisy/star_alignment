function features = extract_point_features(sph, vol, K, method)
if strcmpi(method, 'polarspec')
    features = polar_spectral_feature(sph, vol, K);
elseif strcmpi(method, 'polarvec')
    features = polar_vec(sph, vol, K);
end
% if length(vol) > 1000
%     features = features(vol > prctile(vol,50), :);
% end
end


function features = polar_vec(sph, vol, K)
% This function returns (rho, theta, vol) of a point
[rho, theta, vol] = extract_polar_features(sph, vol, K);
[rho, ri] = sort(rho, 2);
for i = 1:size(sph,1)
    theta(i,:) = theta(i, ri(i,:));
    theta(i,:) = theta(i,:) - theta(i,1);
    vol(i,:) = vol(i, ri(i,:));
end
% rho = rho / std(rho(:));
theta = (theta + pi) / 2/pi / 180;
vol = bsxfun(@times, vol, 1./sum(vol, 2)) / 180;
features = [rho, theta, vol];
end


function features = polar_spectral_feature(sph, vol, K)
[dist_mat, angular_features, vol_mat] = extract_polar_features(sph, vol, K);
fx = -pi:3*pi/180:pi;
features = zeros(size(sph, 1), length(fx));
for j = 1:K
    sigma = 2.5*exp(-dist_mat(:,j)*100) + .04;
    tmp = bsxfun(@minus, fx, angular_features(:, j)).^2;
    tmp = exp(-bsxfun(@times, tmp, .5./sigma.^2));
    tmp = bsxfun(@times, tmp, 1./sigma.*(vol_mat(:, j).*dist_mat(:,j).^2));
    features = features + tmp;
%     features = features + bsxfun(@times, tmp, vol_mat(:,j));
%     features = features + bsxfun(@times, tmp, dist_mat(:,j));
end
features = bsxfun(@times, features, 1./sqrt(sum(features.^2, 2)));
% features = bsxfun(@times, features, 1./max(features, [], 2));
end


