% gsolve.m − Solve for imaging system response function
% Given a set of pixel values observed for several pixels in several
% images with different exposure times, this function returns the
% imaging system’s response function g as well as the log film irradiance
% values for the observed pixels.
% 
% Assumes:
%
% Zmin = 0
% Zmax = 255
%
% Arguments:
%
% Z(i,j) is the pixel values of pixel location number i in image j
% B(j) is the log delta t, or log shutter speed, for image j
% l is lamdba, the constant that determines the amount of smoothness
% w(z) is the weighting function value for pixel value z
% n is the nb of bits
% Returns:
%
% g(z) is the log exposure corresponding to pixel value z
% lE(i) is the log film irradiance at pixel location i

function [g,lE]= gsolve(Z,B,l,w,n)
    size_Z = size(Z);
    num_pixels = size_Z(1);
    num_images = size_Z(2);
    
    A = zeros(num_pixels*num_images + n, n + num_pixels);
    b = zeros(size(A,1), 1);
    
    %% 1. Data fitting
    k = 1;
    for i=1:num_pixels
        for j=1:num_images
            val = Z(i,j) + 1; % MATLAB is 1-indexed
            wij = w(val);
            A(k, val) = wij; 
            A(k, n + i) = -wij; 
            b(k, 1) = wij * B(j);
            k = k + 1;
        end
    end

    %% 2. Constraint (Force mid-point to 0)
    mid_index = floor(n/2) + 1;
    A(k, mid_index) = 1;
    b(k, 1) = 0;
    k = k + 1;

    %% 3. Smoothing
    for i=1:n-2
        A(k, i) = l * w(i+1);
        A(k, i+1) = -2 * l * w(i+1);
        A(k, i+2) = l * w(i+1);
        k = k + 1;
    end

    x = A\b;
    g = x(1:n);
    lE = x(n+1:end);
end