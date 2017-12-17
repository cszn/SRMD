function [im] = im_pad(im)

[w,h,~] = size(im);

if mod(w,2)==1
    im = cat(1,im, im(end,:,:)) ;
end
if mod(h,2)==1
    im = cat(2,im, im(:,end,:)) ;
end

end

