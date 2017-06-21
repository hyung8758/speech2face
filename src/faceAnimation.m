% faceAnimation
% It shows the face parameters as an animation.
% Use this function for the sanity check of the extracted data.
% Input 'data' is structure that was extracted from data4train function.
% 06.15.17
% Hyungwon Yang.
% load a 'data'(structure format) on your workspace named 'face_data'

% get the face parameters.
face = face_data.face_pts;
wave = face_data.wave;
srate = face_data.wave_srate;
f_len = size(face,2);
imrate = f_len/(length(wave)/srate);

%% default position.

ax = plot(face(1:66,1),face(67:end,1),'o');
title('Face Animation','fontsize',20)
axis([-3 3 -3 3])
%hold on;

player = audioplayer(wave, srate);
S.imageRate = 1/imrate;
S.ax = ax;
S.face = face;
S.curIdx = 1;

set(player, 'UserData', S);
set(player, 'TimerFcn', @imageCallback);
set(player, 'TimerPeriod', S.imageRate);
play(player);


function imageCallback(new, eventdata)
    % retrieve
    S = get(new, 'UserData'); 
    idx = S.curIdx;
    idx = idx + 1;
    points = S.face(:,idx);

    % change plot values.
    set(S.ax, 'XData', points(1:66,1),'YData',points(67:end,1))
    S.curIdx = idx;

    % save
    set(new, 'UserData',S); 
end