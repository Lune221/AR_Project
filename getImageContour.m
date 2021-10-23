function [image] = getImageContour(image)
%getImageContour Summary of this function goes here
    %   Detailed explanation goes here
    imageName = "myfilename.png";
    
    image_gray = rgb2gray(image);
    BW = imbinarize(image_gray);

    [B,L] = bwboundaries(BW,'noholes');
    %imshow(label2rgb(L, @jet, [.5 .5 .5]))
    %hold on
    for k = 1:length(B)
       boundary = B{k};
       plot(boundary(:,1), boundary(:,2), 'r', 'LineWidth', 4)
       hold on
    end
    hold off
    set(gca,'visible','off')
    imwrite(getframe(gca).cdata, imageName)
    set(gca,'visible','off')
    
    image = imread(imageName);
    
end

