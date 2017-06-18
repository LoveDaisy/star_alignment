function img = align_average(img_store, feature_store, ref_index, match_function, varargin)
% This function helps align images according to their ground objects
%
% INPUT
%   img_store:      cell array, image type.
%   feature_store:  cell array
%   ref_index:      integer
%   match_function: function handler
% PROPERTIES
%
% OUTPUT
%   img:        m-by-n, double image type

% Some assertion
for i = 1:length(feature_store)
    assert(isfield(feature_store{i}, 'pts'));
end
assert(iscell(img_store));      % TODO

% Handle paramters
parameters = handle_parameters(varargin);

% Info of ref image
img_ref = img_store{ref_index};
ref_gray = rgb2gray(img_ref);
if isinteger(img_ref)
    img_ref = double(img_ref) / double(intmax(class(img_ref)) - 1);
end
img_size = size(img_ref);
ref = imref2d(img_size(1:2), [1, img_size(2)], [1, img_size(1)]);
mean_img = img_ref;

feature_ref = feature_store{ref_index};

% Do averaging
img_num = 1;
for i = 1:length(img_store)
    if i == ref_index
        continue
    end
    
    img = img_store{i};
    if isinteger(img)
        img = double(img) / double(intmax(class(img)) - 1);
    end
    img_num = img_num + 1;
    
    % Extract features of this image
    feature = feature_store{i};
    
    pair_idx = match_function(feature, feature_ref);
    fprintf('Estimating transfrom from image #%d to #%d...\n', i, ref_index);
    tf = find_transform(feature.pts, feature_ref.pts, pair_idx);
    
    if parameters.CheckFeatures
        img_gray = rgb2gray(img);
        figure(2); clf;
        showMatchedFeatures(ref_gray, img_gray, ...
            feature.pts(pair_idx(:,1), :), feature_ref.pts(pair_idx(:,2), :));
        fprintf('Press any key to continue...\n');
        pause;
    end
    
    img_tf = imwarp(img, tf, 'OutputView', ref);
    mean_img = mean_img / img_num * (img_num - 1) + img_tf / img_num;
    
    if parameters.ShowImage
        figure(1); clf;
        imshow(mean_img);
    end
end
img = mean_img;

end


function parameters = handle_parameters(varin)
% Helper function.
% This function handles the parameters.

parameters.CheckFeatures = false;
parameters.ShowImage = false;
if ~isempty(varin)
    if mod(length(varin), 2) ~= 0
        error('Wrong number of arguments!');
    end
    for i = 1:length(varin)/2
        key = varin{i*2-1};
        value = varin{i*2};
        switch lower(key)
            case 'checkfeatures'
                if islogical(value)
                    parameters.CheckFeatures = value;
                else
                    error('CheckFeatures must be a logical value!');
                end
            case 'showimages'
                if islogical(value)
                    parameters.ShowImages = value;
                else
                    error('ShowImages must be a logical value!');
                end
            otherwise
                error('Wrong property name!');
        end
    end
end
end
