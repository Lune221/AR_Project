image = imread('voiture.jpg');
imageContour = getImageContour(image);

% bw = sum(imageContour,3) > 750;        % a binary image to overlay
% mask = cast(bw, class(imageContour));  % ensure the types are compatible
% imageContour = imageContour .* repmat(mask, [1 1 3]);  % apply the mask
% figure
% imshow(imageContour)
imshow(imageContour)
