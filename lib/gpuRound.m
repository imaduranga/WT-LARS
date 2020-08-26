function [x] = gpuRound(x,decimals)
%GPUROUND Summary of this function goes here
%   Detailed explanation goes here

if isa(x,'gpuArray')  
    x = round(x*10^decimals)*10^-decimals;
else
    x = round(x,decimals);
end

end

