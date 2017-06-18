function [files, digits, img_info] = get_image_files(img_path, regexp_pattern)
% This function scans a directory and get all valid image files infomation.
%
% INPUT
%   img_path:           string
%   regexp_pattern:     string
% OUTPUT
%   files:      file structure array
%   digits:     number array
%   img_info:   image info structure cell array

orig_state = warning;
warning('off','all');

files = dir(img_path);
digits = zeros(length(files), 1);
img_info = cell(length(files), 1);

is_valid_file = true(length(files), 1);
fprintf('Checking all image files...\n');
for i = 1:length(files)
    d = regexp(files(i).name, regexp_pattern, 'tokens');
    if isempty(d)
        is_valid_file(i) = false;
    else
        digits(i) = str2double(d{1});
        img_info{i} = imfinfo([img_path, files(i).name]);
    end
end

files = files(is_valid_file);
digits = digits(is_valid_file);
img_info = img_info(is_valid_file);

[digits, idx] = sort(digits);
files = files(idx);
img_info = img_info(idx);

warning(orig_state);
end