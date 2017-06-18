% This script aligns the star to a given reference image,
% and average them to get a denoised image.

clear; clc; close all;

img_path = '../python/';
out_path = '../python';
regexp_pattern = '\<IMG_(\d{4})_0\.TIF$';

[files, digits, img_info] = get_image_files(img_path, regexp_pattern);
default_focal_length = 20;

start_digit = 1561;
end_digit = 1562;
ref_digit = 1561;

file_idx = (digits >= start_digit & digits <= end_digit);
files = files(file_idx);
digits = digits(file_idx);
ref_idx = find(abs(digits - ref_digit) < 0.5);

img_num = length(files);

img_store = cell(img_num, 1);
feature_store = cell(img_num, 1);

mask = [];
% mask = imread([img_path, 'IMG_9144_mask.png']);
% mask = mask(:,:,1) > 1;
k = 1;
for i = 1:length(files)
    if digits(i) < start_digit || digits(i) > end_digit
        continue;
    end
    fprintf('Reading image %s...\n', files(i).name);
    img = imread([img_path, files(i).name]);
    if isinteger(img)
        img = double(img) / double(intmax(class(img)) - 1);
    end
    img_store{k} = img;
    
    fprintf('Converting image to gray...\n');
    img_gray = rgb2gray(img_store{k});
    
    fprintf('Detecting features...\n');
    info = img_info{i};
    if isfield(info, 'DigitalCamera') && isfield(info.DigitalCamera, 'FocalLength')
        f = info.DigitalCamera.FocalLength;
    else
        f = 16;
    end
    feature = extract_star_feature(img_gray, mask, f);
    feature_store{k} = feature;
    
    k = k + 1;
end
    
%%
fprintf('Denoising...\n');
img = align_average(img_store, feature_store, ref_idx, @find_initial_match);

figure(1); clf;
imshow(img);
drawnow;

fprintf('Writing image...\n');
imwrite(uint16(img*65535), [out_path, files(ref_idx).name, '_star_avg.tif']);


