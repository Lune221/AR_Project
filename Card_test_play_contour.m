%% Fix number of matching points
MATCHING_POINTS_NUMBER = 7;

%% Load Refrences image, detect SURF points and extract descriptors

referenceImage = imread("couverture.jpeg");

%% Detect and extract SURF features
referenceImageGray = rgb2gray(referenceImage);
referencePts = detectSURFFeatures(referenceImageGray);

referenceFeatures = extractFeatures(referenceImageGray, referencePts);

%% Initialise replacement image

image = imread('me.png');
imageContour = getImageContour(image);
bw = sum(imageContour,3) > 700;        % a binary image to overlay
mask = cast(bw, class(imageContour));  % ensure the types are compatible
imageContour = imageContour .* repmat(mask, [1 1 3]);  % apply the mask

%% Prepare video input from webcam

camera = webcam();

% Capture one frame to get its size.
cameraFrame = snapshot(camera);
frameSize = size(cameraFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

runLoop = true;
fix = false;
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
        if fix == false
            
            % Store the SURF points that were matched
            matchedCameraPts = cameraPts(idxPairs(:,1));
            matchedReferencePts = referencePts(idxPairs(:,2));
            try
                %% Get geometric tansformation between reference Image and webcam Frame

                [referenceTransform, inlierReferencePts, inlierCameraPts] = estimateGeometricTransform(matchedReferencePts, matchedCameraPts, 'Similarity');

                %% Rescale Replacement Video frame

                % Get replacement and reference dimensions
                repDims = size(imageContour(:, :, 1));
                refDims = size(referenceImage);

                % Find transformation that scales video frame to replacement image size
                % preserving aspect ratio

                scaleTransform = findScaleTransform(refDims, repDims);

                outputView = imref2d(size(referenceImage));
                imageContourScaled = imwarp(imageContour, scaleTransform, 'OutputView', outputView);

                %% Apply estimated geometric transform to scaled replacement video frame
                outputView = imref2d(size(cameraFrame));
                imageContourTransformed = imwarp(imageContourScaled, referenceTransform, 'OutputView', outputView);

                %% Insert transformed Video Frame into webcam frame

                alphaBlender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

                mask = imageContourTransformed(:, :, 1) | imageContourTransformed(:,:,2) | imageContourTransformed(:,:,3) > 0;

                outputFrame = step(alphaBlender, cameraFrame, imageContourTransformed, mask);

                step(videoPlayer, outputFrame);
                fix = true;
            catch ME
                warning("An error has ocurred");
                rethrow(ME);
            end
        else
           outputFrame = step(alphaBlender, cameraFrame, imageContourTransformed, mask);
           step(videoPlayer, outputFrame);
        end
        
    elseif fix == true
        outputFrame = step(alphaBlender, cameraFrame, imageContourTransformed, mask);
        step(videoPlayer, outputFrame);
    else
        warning("Can't see the reference image in the camera frame");
        step(videoPlayer, cameraFrame);
    end

    runloop = isOpen(videoPlayer);
end
clearvars