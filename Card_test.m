%% Load Refrences image, detect SURF points and extract descriptors

referenceImage = imread("reference.jpeg");

%% Detect and extract SURF features
referenceImageGray = rgb2gray(referenceImage);
referencePts = detectSURFFeatures(referenceImageGray);

referenceFeatures = extractFeatures(referenceImageGray, referencePts);

%% Display SURF features for a reference Image

figure;
imshow(referenceImage), hold on;
plot(referencePts.selectStrongest(50));

%% Initialise replacement video

video = vision.VideoFileReader('Jamono.mp4', 'VideoOutputDataType', 'uint8');

% Skip the first fex black frames
for k = 1:30
    step(video);
end

%% Prepare video input from webcam

camera = webcam();
%set(camera, 'Resolution', '640x480');

%% Detect SURF features in webcam frame

cameraFrame = snapshot(camera);

cameraFrameGray = rgb2gray(cameraFrame);
cameraPts = detectSURFFeatures(cameraFrameGray);

figure(1)
imshow(cameraFrame), hold on;
plot(cameraPts.selectStrongest(50));

%% Try to match the reference Image and the camera frame features

cameraFeatures = extractFeatures(cameraFrameGray, cameraPts);
idxPairs = matchFeatures(cameraFeatures, referenceFeatures);

% Store the SURF points that were matched
matchedCameraPts = cameraPts(idxPairs(:,1));
matchedReferencePts = referencePts(idxPairs(:,2));

figure(1)
showMatchedFeatures(cameraFrame, referenceImage, matchedCameraPts, matchedReferencePts, 'Montage');

%% Get geometric tansformation between reference Image and webcam Frame

[referenceTransform, inlierReferencePts, inlierCameraPts] = estimateGeometricTransform(matchedReferencePts, matchedCameraPts, 'Similarity');

% Show the inliers of the estimated geometric transformation
figure(1)
showMatchedFeatures(cameraFrame, referenceImage, inlierCameraPts, inlierReferencePts, 'Montage');

%% Rescale Replacement Video frame

% Load replacement video frame
videoFrame = step(video);

% Get replacement and reference dimensions
repDims = size(videoFrame(:, :, 1));
refDims = size(referenceImage);

% Find transformation that scales video frame to replacement image size
% preserving aspect ratio

scaleTransform = findScaleTransform(refDims, repDims);

outputView = imref2d(size(referenceImage));
videoFrameScaled = imwarp(videoFrame, scaleTransform, 'OutputView', outputView);

figure(1)
imshowpair(referenceImage, videoFrameScaled, 'Montage');

%% Apply estimated geometric transform to scaled replacement video frame
outputView = imref2d(size(cameraFrame));
videoFrameTransformed = imwarp(videoFrameScaled, referenceTransform, 'OutputView', outputView);

figure(1)
imshowpair(cameraFrame, videoFrameTransformed, 'Montage');

%% Insert transformed Video Frame into webcam frame

alphaBlender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

mask = videoFrameTransformed(:, :, 1) | videoFrameTransformed(:,:,2) | videoFrameTransformed(:,:,3) > 0;

outputFrame = step(alphaBlender, cameraFrame, videoFrameTransformed, mask);

figure(1);
imshow(outputFrame);

%% Initialize point Tracker

pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
initialize(pointTracker, inlierCameraPts.Location, cameraFrame);

% Display the points being used for tracking
trackingMarkers = insertMarker(cameraFrame, inlierCameraPts.Location, 'Size', 7, 'Color', 'yellow');
figure(1);
imshow(trackingMarkers);

%% Track points to next frame

% Store previous frame just for visual comparison
prevCameraFrame = cameraFrame;

% Get next ebcam frame
cameraFrame = snapshot(camera);

% Find newly tracked points
[trackedPoints, isValid] = step(pointTracker, cameraFrame);
% Use only the locations that have been reliably tracked
newValidLocations = trackedPoints(isValid, :);
oldValidLocations = inlierCameraPts.Location(isValid, :);
figure(1)
imshow(cameraFrame), hold on;

%% Estimate geometric transformation between two frames

if (nnz(isValid) >=2) % Must have at least 2 tracked points between frames
    [trackingTransform, oldInlierLocations, newInlierLocations] = estimateGeometricTransform(oldValidLocations, newValidLocations, 'Similarity');
end

% Show the valid of the geometric transformation
figure(1)
showMatchedFeatures(prevCameraFrame, cameraFrame, oldInlierLocations, newInlierLocations, 'Montage');

% Reset Point Tracker for tracking in next frame
setPoints(pointTracker, newValidLocations);

%% Accumulate geometric transformations from reference to current frame

trackingTransform.T = referenceTransform.T * trackingTransform.T;

%% Rescale new replacement video frame

repframe = step(video);
outputView = imref2d(size(referenceImage));
videoFrameScaled = imwarp(videoFrame, scaleTransform, 'OutputView', outputView);

figure(1)
imshowpair(referenceImage, videoFrameScaled, 'Montage');

%% Apply total geometric transformation to new replacement video frame

outputView = imref2d(size(cameraFrame));
videoFrameTransformed = imwarp(videoFrameScaled, trackingTransform, 'OutputView', outputView);

figure(1)
imshowpair(cameraFrame, videoFrameTransformed, 'Montage');

%% Insert transformed replacement frame into webcam input

mask = videoFrameTransformed(:, :, 1) | videoFrameTransformed(:, :, 2) | videoFrameTransformed(:, :, 3) > 0;

outputFrame = step(alphaBlender, cameraFrame, videoFrameTransformed, mask);

figure(1)
imshow(outputFrame);

%% Clean all
release(video);
delete(camera);