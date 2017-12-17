function [im] = im_crop(im,w,h)


if mod(w,2)==1
    im = im(1:end-1,:,:);
end
if mod(h,2)==1
    im = im(:,1:end-1,:);
end

end

