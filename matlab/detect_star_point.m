function pts = detect_star_point(img, mask, method, varargin)
% This function detects the stars in an image
% INPUT
%   img:    m-by-n, double, gray scale, normalized between 0 and 1
%   mask:   m-by-n, boolean, or empty
%   varargin:   variable-length input
% OUTPUT
%   pts:    struct array with fields,
%               1. location, [x, y]
%               2. intensity
%               3. area

assert(length(size(img)) == 2);
assert(isfloat(img));
assert(islogical(mask) || isempty(mask));
assert(all(size(img) == size(mask)) || isempty(mask));

if isempty(mask)
    mask = true(size(img));
end

if ~isempty(varargin)
    assert(isnumeric(varargin{1}));
    s = varargin{1};
    img = imfilter(img, fspecial('gaussian', 3*s*[1, 1], s));
end
img = (img - mean(img(mask))) / (max(img(mask)) - min(img(mask)));

if strcmpi(method, 'wavelet')
    [img_bw, img_gray] = wavelet_method(img, mask);
elseif strcmpi(method, 'conv')
    [img_bw, img_gray] = conv_method(img, mask);
end

stats = regionprops(bwmorph(img_bw, 'open'), img_gray, ...
    'WeightedCentroid', 'Area', 'Eccentricity', 'MeanIntensity');

weighted_centroid = cat(1, stats.WeightedCentroid);
area = cat(1, stats.Area);
eccentricity = cat(1, stats.Eccentricity);
intensity = cat(1, stats.MeanIntensity);
ind = area > 5 & area < 200 & eccentricity < .9;
ind = ind & area > prctile(area, 20) & intensity > prctile(intensity, 20);

location = cell(sum(ind), 1);
weighted_centroid = weighted_centroid(ind, :);
for i = 1:sum(ind)
    location{i} = weighted_centroid(i, :);
end
pts = struct('location', location, ...
    'intensity', arrayfun(@(x){x}, intensity(ind)), ...
    'area', arrayfun(@(x){x}, area(ind)));

end


% =============================================================================
% convolution method
% =============================================================================
function [bw, img0] = conv_method(img, mask)
h = fspecial('gaussian', 10 * [1, 1], 1.2);
h = h - mean(h(:));
img = img - mean(img(:));

img0 = imfilter(img, h) .* mask;
img0(img0 < 0) = 0;
img0 = img0 / max(img0(:));

bw = img0 > .1;
end


% =============================================================================
% wavelet method
% =============================================================================
function [bw, img_rec] = wavelet_method(img, mask)
method = 'db8';
[C, S] = wavedec2(img, 6, method);
C(1:S(1,1)*S(1,2)) = 0;
C(end-sum(S(end-1,1).*S(end-1,2)*3)+1:end) = 0;
img_rec = waverec2(C, S, method) .* mask;
bw = img_rec > prctile(img_rec(mask), 99.5);
end
