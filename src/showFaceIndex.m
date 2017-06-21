function showFaceIndex(data,varargin)
% Show the face index by plotting the parameter point by point.
% Input 'data' is structure that was extracted from data4train function.
% 06.15.17
% Hyungwon Yang.

spd = 1;
if nargin > 1
    spd = varargin{1};
    if ischar(spd)
        error('2nd input should be a number(float or integer).')
    end
end
if nargin > 2
    error('The number of input arguments should not be greater than 2.')
end

% get the face parameters.
face = data.param;

% plot the face with index.
for i = 1:length(face{1})
    plot(face{1}(i,1),face{1}(i,2),'o')
    hold on;
    title(['index: ',num2str(i)],'fontsize',20)
    axis([-3 3 -3 3])
    shg
    pause(spd)
end