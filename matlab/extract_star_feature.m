function feature = extract_star_feature(img, mask, f)
% This function extracts features of all stars
%
% INPUT
%   img:        m-by-n, gray scale image
%   mask:       m-by-n, logical
%   f:          scalar, effective focal length
% OUTPUT
%   feature:    struct

sigma = 3;
K = 15;

fprintf('Detecting star points...\n');
pts_info = detect_star_point(img, mask, 'wavelet', sigma);
% pts_info = detect_star_point(img, mask, 'conv', sigma);
fprintf('  done! Total %d star points detected!\n', size(pts_info, 1));

vol = cat(1, pts_info.intensity) .* cat(1, pts_info.area);
pts = cat(1, pts_info.location);
sph = convert_coord_img_sph(pts, size(img), f);

fprintf('Extracting feature of each star point...\n');
pf = extract_point_features(sph, vol, K, 'polarspec');
feature = struct('polar_feature', pf, 'pts', pts, 'sph', sph);
end
