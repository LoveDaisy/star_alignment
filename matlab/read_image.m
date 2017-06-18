function [img, index_info, info] = read_image(img_path, name_pattern, index_info)
% This function helps read image from a given directory
%
% INPUT
%   img_path:       string
%   name_pattern:   struct, include field of 'regexp' and 'print'
%   index_info:     struct, include field of 'last_index',
%                   'ref_index', 'ref_digits',
%                   'start_index', 'start_digits', 'end_index', 'end_digits'
%                   'method' ('single', 'continous'),
%                   'image_name'
% OUTPUT
%   img:            m-by-n, or m-by-n-by-3 matrix. normalized to [0, 1]
%   index_info:     struct, // TODO
%   info:           struct, exif info

assert(isfield(index_info, 'last_index'));

img_digits = get_image_digits(img_path, name_pattern.regexp);

% Deal with index
start_index = -1;
end_index = -1;
if isfield(index_info, 'start_index')
    start_index = index_info.start_index;
end
if isfield(index_info, 'end_index')
    end_index = index_info.end_index;
end
if isfield(index_info, 'start_digits')
    start_index = find(img_digits == index_info.start_digits);
end
if isfield(index_info, 'end_digits')
    end_index = find(img_digits == index_info.end_digits);
end
if isempty(start_index)
    start_index = -1;
end
if isempty(end_index)
    end_index = -1;
end

ref_index = -1;
if isfield(index_info, 'ref_index')
    ref_index = index_info.ref_index;
end
if isfield(index_info, 'ref_digits')
    ref_index = find(img_digits == index_info.ref_digits);
end
if isempty(ref_index)
    ref_index = -1;
end

current_index = index_info.last_index + 1;
if current_index < start_index
    current_index = start_index;
end
if current_index < 0
    current_index = 0;
end
if isfield(index_info, 'method') && strcmpi(index_info.method, 'continous')
    if ref_index > 0 && ref_index == current_index
        current_index = current_index + 1;
    end
else
    if ref_index > 0
        current_index = ref_index;
    else
        error('ERROR! Reference image is not found!');
    end
end
if current_index > size(img_digits,1) || (end_index > 0 && current_index > end_index)
    img = [];
    info = [];
    index_info.last_index = end_index;
    return
end

index_info.last_index = current_index;

file_name = sprintf(name_pattern.print, img_digits(current_index));
fprintf('Reading image %s\n', file_name);

img = imread([img_path, file_name]);
max_value = intmax(class(img));
img = double(img) / double(max_value);

info = imfinfo([img_path, file_name]);
index_info.image_name = file_name;

end


function img_digits = get_image_digits(img_path, regexp_pattern)
% Get all image digits in a directory

files = dir(img_path);
img_digits = nan(length(files), 1);
for i = 1:length(files)
    d = regexp(files(i).name, regexp_pattern, 'tokens');
    if isempty(d)
        continue;
    end
    img_digits(i) = str2double(d{1});
end
img_digits = unique(img_digits(~isnan(img_digits)));

end
