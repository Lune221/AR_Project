%% Fix number of matching points
MATCHING_POINTS_NUMBER = 7;

%% Load Refrences image, detect SURF points and extract descriptors

referenceImage = imread("reference.jpeg");

%% Detect and extract SURF features
referenceImageGray = rgb2gray(referenceImage);
referencePts = detectSURFFeatures(referenceImageGray);

referenceFeatures = extractFeatures(referenceImageGray, referencePts);

%% Initialise replacement video

video = vision.VideoFileReader('Jamono.mp4', 'VideoOutputDataType', 'uint8');
% Skip the first fex black frames
for k = 1:30
    step(video);
end

%% Prepare video input from webcam

camera = webcam();

% Capture one frame to get its size.
cameraFrame = snapshot(camera);
frameSize = size(cameraFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

runLoop = true;
isValid = 0;
frameCount = 0;

while runLoop && frameCount < 400
   %% Detect SURF features in webcam frame
    frameCount = frameCount + 1;
    cameraFrame = snapshot(camera);

    cameraFrameGray = rgb2gray(cameraFrame);
    cameraPts = detectSURFFeatures(cameraFrameGray);
    
    %% Try to match the reference Image and the camera frame features

    cameraFeatures = extractFeatures(cameraFrameGray, cameraPts);
    idxPairs = matchFeatures(cameraFeatures, referenceFeatures);
    idxPairsSize = size(idxPairs(:, 2), 1);
    
    if idxPairsSize > MATCHING_POINTS_NUMBER
        % Store the SURF points that were matched
        matchedCameraPts = cameraPts(idxPairs(:,1));
        matchedReferencePts = referencePts(idxPairs(:,2));
        try
            %% Get geometric tansformation between reference Image and webcam Frame

            [referenceTransform, inlierReferencePts, inlierCameraPts] = estimateGeometricTransform(matchedReferencePts, matchedCameraPts, 'Similarity');

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

            %% Apply estimated geometric transform to scaled replacement video frame
            outputView = imref2d(size(cameraFrame));
            videoFrameTransformed = imwarp(videoFrameScaled, referenceTransform, 'OutputView', outputView);

            %% Insert transformed Video Frame into webcam frame

            alphaBlender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

            mask = videoFrameTransformed(:, :, 1) | videoFrameTransformed(:,:,2) | videoFrameTransformed(:,:,3) > 0;

            outputFrame = step(alphaBlender, cameraFrame, videoFrameTransformed, mask);

            step(videoPlayer, outputFrame);
        catch ME
            warning("An error has ocurred");
        end
    else
        warning("Can't see the reference image in the camera frame");
        step(videoPlayer, cameraFrame);
    end

    runloop = isOpen(videoPlayer);
end
release(video)
delete(camera)