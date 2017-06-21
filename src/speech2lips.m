% speech2lips
%                               Hyungwon Yang

function speech2lips()
% Set your trained model.
train_model = './data/default/sample_net.mat';


S.model = train_model;
S.fh = figure('units','pixels',...  % figure shape
    'position',[250 250 800 600],...
    'numbertitle','off',...
    'menubar','none',...
    'name','Real-time Estimator',...
    'renderer', 'painters',...
    'resize','off');
S.ax_art = axes('units','pixels',...    % plotting axes
    'position',[200,200,450,350]);
S.pb_start = uicontrol('style','push',... % start button
    'units','pixels',...
    'position',[280 60 100 50],...
    'string','Start',...
    'value',0,...
    'fontsize',15);
S.pb_stop = uicontrol('style','push',...  % stop button
    'units','pixels',...
    'position',[480 60 100 50],...
    'string','Stop',...
    'value',0,...
    'enable','off',...
    'fontsize',15);

% must update callback functions at this point
% if not, S will not contain timerObj, recordObj and parameters
set(S.pb_start,'callback',{@pb_start}) 
set(S.pb_stop,'callback',{@pb_stop})

% basic settings
set_parameters(S);

% axes setting
set_axes(S);


% recording object
S.recordObj = audiorecorder(S.srate,24,1); % nbit=24, channel=1 (mono)
set(S.recordObj, 'TimerPeriod', 0.1); % time interval for each TimerFcn

% timer object (should be the last one to define)
S.timerObj = timer('TimerFcn',{@drawGUI},...
    'Period',S.timeInterval,...
    'ExecutionMode','fixedRate');
set(S.fh,'UserData',S) % save timerObj in UserData

% must update callback functions at this point
% if not, S will not contain timerObj, recordObj and parameters
set(S.pb_start,'callback',{@pb_start}) 
set(S.pb_stop,'callback',{@pb_stop})


%-- pushbotton: start plotting
    function pb_start(varargin)
        set(S.pb_start,'enable','off')
        set(S.pb_stop,'enable','on')
        S.recEndPoint = 0; % once stop button is pressed, tmpAudio is flushed. That's why!
        S.lastPosition = 1; % lastPosition should be back to 1 if resumed after a pause
        set(S.fh,'UserData',S) % update S; if not, S.recEndPoint won't be 0

        record(S.recordObj); % start recording
        pause(.5) % quick pause for recording object to initiate
        start(S.timerObj); % start timer object
    end

%-- pushbotton: stop plotting
    function pb_stop(varargin)
        set(S.pb_stop,'enable','off')
        set(S.pb_start,'enable','on')
        
        stop(S.timerObj)
        stop(S.recordObj)
    end

%-- drawing samples on the axes
    function drawGUI(varargin)
        % This function calls drawGUIsgram, drawGUIwave
        S = get(S.fh,'UserData'); % retrieve S from UserData
        S.tmpAudio = getaudiodata(S.recordObj); % tmpAudio is adding up previous data
%         S.spectrogramBuffer = get(S.sgramHandle,'cdata'); % get initial spectrogram data
        S.recCurrentPoint = S.recEndPoint + 1; % recording start point
        S.recEndPoint = length(S.tmpAudio); % recording end point
        S.frameEndPoint = S.recCurrentPoint + S.windowLengthInSample; % end point for each frame
%         set(S.tx,'string',sprintf('numFrame = %d',S.frameCount))
        
        % draw spectrogram
        drawGUIsgram(varargin);
        
        S.frameCount = S.frameCount+1;
        
        set(S.fh,'UserData',S) % update S in UserData!
    end

%-- spectrogram & articulation
    function drawGUIsgram(varargin)
        nFrames = 0;
        while S.lastPosition+S.frameShiftInSample_sgram+S.windowLengthInSample < S.recEndPoint
            % Whenever recordObj is called, samples will be obtained such
            % that for that samples, it should be windowed as much as
            % possible to maximize the use of information. 
            nFrames = nFrames + 1;
            currentIndex = S.lastPosition + S.frameShiftInSample_sgram;
            x = S.tmpAudio(currentIndex + (0:S.windowLengthInSample-1));
%             tmpSpectrum = 20*log10(abs(fft(x.*S.windowFunction,S.fftLength)));
%             S.fftBuffer(:,nFrames) = tmpSpectrum; % S.fftBuffer will be overwritten
            S.lastPosition = currentIndex;
        end
        mfcc = get_mfcc(x, S.srate);
        pred = forward(mfcc); % (example x feature)
        
%         tmpSgram = S.fftBuffer(:,1:nFrames);
%         if S.maxLevel < max(tmpSgram(:))
%             S.maxLevel = max(tmpSgram(:));
%         else
%             S.maxLevel = max(-100,S.maxLevel*0.998);
%         end
%         tmpSgram = 62*max(0,(tmpSgram-S.maxLevel)+S.dynamicRange)/S.dynamicRange+1;
%         
%         S.spectrogramBuffer(:,1:end-nFrames) = S.spectrogramBuffer(:,nFrames+1:end); % back portion to front
%         S.spectrogramBuffer(:,end-nFrames+1:end) = tmpSgram(1:length(S.freqAxis_sgram),:); % update back portion
%         set(S.sgramHandle,'cdata',S.spectrogramBuffer)
        
        % draw articulation
        plot(S.ax_art,pred(1:18),pred(19:36),...
            'ob','MarkerSize',8,...
            'MarkerFaceColor','k') % (example x feature)
        hold on
        pad=18;
        
%         line(S.ax_art,pred([1,14,13,12,5]),pred([pad+1,pad+14,pad+13,pad+12,pad+5]),'color',[0 0 1],'linewidth',2)
%         line(S.ax_art,pred([1,9,10,11,5]),pred([pad+1,pad+9,pad+10,pad+11,pad+5]),'color',[0 0 1],'linewidth',2)
%         line(S.ax_art,pred([1,8,7,6,5]),pred([pad+1,pad+8,pad+7,pad+6,pad+5]),'color',[0 0 1],'linewidth',2)
%         line(S.ax_art,pred([1,2,3,4,5]),pred([pad+1,pad+2,pad+3,pad+4,pad+5]),'color',[0 0 1],'linewidth',2)
        line(S.ax_art,pred([1,2,3,4,5,6,7]),pred([pad+1,pad+2,pad+3,pad+4,pad+5,pad+6,pad+7]),'color',[0 0 1],'linewidth',2)
        line(S.ax_art,pred([1,13,14,15,7]),pred([pad+1,pad+13,pad+14,pad+15,pad+7]),'color',[0 0 1],'linewidth',2)
        line(S.ax_art,pred([1,18,17,16,7]),pred([pad+1,pad+18,pad+17,pad+16,pad+7]),'color',[0 0 1],'linewidth',2)
        line(S.ax_art,pred([1,12,11,10,9,8,7]),pred([pad+1,pad+12,pad+11,pad+10,pad+9,pad+8,pad+7]),'color',[0 0 1],'linewidth',2)
         
        hold(S.ax_art,'on')
%         S.pal = plot(S.ax_art,S.meanPalates(:,1),S.meanPalates(:,2),'k-');
%         S.pha = plot(S.ax_art,S.meanPharynx(:,1),S.meanPharynx(:,2),'k-');
        title(S.ax_art,'speech2lips','fontsize',20)
        grid(S.ax_art,'minor')
        hold(S.ax_art,'off')        
        xlim(S.ax_art,[-1.5 1.5])
        ylim(S.ax_art,[-1.5 0.5])
        hold(S.ax_art,'off')
        set(S.fh,'UserData',S) % update S in UserData!
    end

    function mfcc = get_mfcc(sig, srate)
        Tw = 25;           % analysis frame duration (ms)
        Ts = 10;           % analysis frame shift (ms)
        alpha = 0.97;      % preemphasis coefficient
        R = [ 300 3700 ];  % frequency range to consider
        M = 20;            % number of filterbank channels
        C = 13;            % number of cepstral coefficients
        L = 22;            % cepstral sine lifter parameter
        [mfcc_tmp,~,~,~] = ...
            mfcc_rev( sig, srate, Tw, Ts, alpha, @hamming, R, M, C, L );
        % get 1st and 2nd derivatives
        deriv_1 = [zeros(size(mfcc_tmp,1),1), diff(mfcc_tmp,1,2)];
        deriv_2 = [zeros(size(mfcc_tmp,1),2), diff(mfcc_tmp,2,2)];
        mfcc = [mfcc_tmp;deriv_1;deriv_2]; 
    end

%-- Forward calculation
    function pred = forward(xdata)
        % xdata: (feature x examples)
        pred = S.model(xdata); % (example x feature)     
    end

%-- parameter setting
    function set_parameters(varargin)
        % this function set basic parameters
        net = load(S.model);
        S.model = net.net;
        S.timeInterval = 0.1; % timer interval in sec
        S.srate = 16000; % sampling rate
        S.windowLengthMS = 80; % in ms
        S.higherFreqLimit = 5000; % max limit for drawing frequencies
        S.fftLength = 2^ceil(log2(S.windowLengthMS*S.srate/1000)); % fft input length e.g. 4096
        S.frameShift_sgram = 1/S.srate*300; % in sec
        S.frameShift_fft = 1/S.srate*300; % in sec
        S.frameShift_wave = 1/S.srate; % in sec
        S.frameShiftInSample_sgram = round(S.frameShift_sgram*S.srate); % e.g. 0.007*44100=>309
        S.frameShiftInSample_fft = round(S.frameShift_fft*S.srate); % e.g. 0.007*44100=>309
        S.windowFunction = nuttallwin(round(S.windowLengthMS*S.srate/1000)); % window function
        S.windowLengthInSample = length(S.windowFunction); % samples in a window e.g. 3528
        S.dynamicRange = 80;
        S.lastPosition = 1;
        S.frameCount = 0; % counting every frame
        S.recEndPoint = 0; % recording end point for tmpAudio in drawGUIsgram function
        S.maxLevel = -100;
    end

%-- axes setting
    function set_axes(varargin)
        % this function set axes before plotting
        tmpfreqAxis = (0:S.fftLength/2)/S.fftLength*S.srate;
        S.plotTimeAxisSize = 4;   % e.g. 4 second x-axis in plot
        S.freqAxis_sgram = tmpfreqAxis(tmpfreqAxis<S.higherFreqLimit); % y-axis frequency range
        S.freqAxis_fft = linspace(0,floor(S.srate/2),S.fftLength/2); % x-axis frequency range in Hz
        
        % face plotting.
        if S.model.outputs{3}.size == 36
            base_pos = load('./data/default/base_position_v1.mat');
        Y = base_pos.base_pos;
        plot(S.ax_art,Y(1:18),Y(19:36),...
            'ob','MarkerSize',8,...
            'MarkerFaceColor','k');
        xlim(S.ax_art,[-1.5 1.5])
        ylim(S.ax_art,[-1.5 0.5])
        hold on
        pad=18;
%         plot(S.ax_art,Y(ptt),Y(pad+ptt),'or');
        line(S.ax_art,Y([1,2,3,4,5,6,7]),Y([pad+1,pad+2,pad+3,pad+4,pad+5,pad+6,pad+7]),'color',[0 0 1],'linewidth',2)
        line(S.ax_art,Y([1,13,14,15,7]),Y([pad+1,pad+13,pad+14,pad+15,pad+7]),'color',[0 0 1],'linewidth',2)
        line(S.ax_art,Y([1,18,17,16,7]),Y([pad+1,pad+18,pad+17,pad+16,pad+7]),'color',[0 0 1],'linewidth',2)
        line(S.ax_art,Y([1,12,11,10,9,8,7]),Y([pad+1,pad+12,pad+11,pad+10,pad+9,pad+8,pad+7]),'color',[0 0 1],'linewidth',2)
        
        title(S.ax_art,'speech2lips','fontsize',20)
        grid(S.ax_art,'minor')
        hold(S.ax_art,'off')
        else
            error('Other data property is not supported.')
        end
    end


end