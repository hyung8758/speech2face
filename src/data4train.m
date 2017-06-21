% data4train.m
% 
% This function preprocesses files in a data directory in order them to be
% ready for speech2mouth(aka speech2face) training.
%
% 06.15.17
% Hyungwon Yang

function data = data4train(data_folder)

% search data folder. sanity check.
param_name = 'parameters.txt';
image_name = 'image';
wave_name = 'sample.pcm';

directory = dir(data_folder);
allNames = { directory.name };

for check = 1:length(allNames)
    % check parameters.txt file.
    if strcmp(allNames(check),param_name)
        fprintf('Parameter file is detected.\n')
        fid = fopen(fullfile(data_folder,param_name));
        T = fgets(fid);
        box_idx = 1;
        line_idx = 1;
        while ischar(T)
            if isempty(regexp(T,'-----','once'))
                tmp_param = split(T,',');
                param{box_idx}(line_idx,1) = str2num(tmp_param{1});
                param{box_idx}(line_idx,2) = -(str2num(tmp_param{2}));
                line_idx = line_idx+1;
            else
                line_idx = 1;
                box_idx = box_idx+1;
            end
            T = fgets(fid);
        end
        if isempty(param)
            error('parameters.txt is empty.')
        end
        fprintf('pts parameters are successfully imported.\n')
        % normalizing parameters.
        for p = 1:length(param)
            if length(param{p}) == 66
                new_param{p} = zscore(param{p});
                % save it to data structure.
                data.param = new_param;
            end
        end
    end

    % check image file.
    if strcmp(allNames(check),image_name)
        img_dir = dir(fullfile(data_folder,image_name));
        img_name = { img_dir.name };
        count = 0;
        for i = 1:length(img_name)
            if isempty(regexp(img_name{i},'^\.'))
                count = count+1;
            end
        end     
        if count == 0
            image = [];
            % save it to data structure.
            data.image = image;
            warning('Images are not found in a directory. Ignore the image process.')
        else
            % importing images.
            image = [];
            % save it to data structure.
            data.image = image;
            fprintf('Images are found but function is not yet built to process them.\n')
        end
    end
    
    % check wave file.
    if strcmp(allNames(check),wave_name)
        fprintf('Wave file is detected.\n')
        fid = fopen(fullfile(data_folder,wave_name));
        tmp_wave = fread(fid,inf,'short');
        fclose(fid);
        % save it to data structure.
        wave = scale_sound(tmp_wave);
        data.wave = wave;
        data.wave_srate = 16000;
        fprintf('Wave file is successfully imported.\n')
    end
end
if length(fields(data)) ~= 4
    error('%s folder is corrupted.',data_folder)
end

% extract mouth information.
fprintf('Extract mouth parameters.\n')
face = data.param;
for i = 1:length(face)
    con = 1;
    for j = 49:66
        mouth{i}(con,1) = face{i}(j,1);
        mouth{i}(con,2) = face{i}(j,2);
        con = con+1;
    end
end
% reorganize mouth and face pts parameters.
for order = 1:size(mouth,2)
    m_vec = [mouth{order}(:,1); mouth{order}(:,2)];
    f_vec = [face{order}(:,1); face{order}(:,2)];
    m_pts(:,order) = m_vec;
    f_pts(:,order) = f_vec;
end

% save it to data structure.
data.mouth_pts = m_pts;
data.face_pts = f_pts;
fprintf('DONE: extracting mouth parameters.\n')

%% extract MFCC
fprintf('Extract mfcc.\n')
srate = 16000;
Tw = 25;           % analysis frame duration (ms)
Ts = 10;           % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [ 300 3700 ];  % frequency range to consider
M = 20;            % number of filterbank channels
C = 13;            % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter

[mfcc_tmp,~,~,~] = ...
    mfcc_rev( data.wave, srate, Tw, Ts, alpha, @hamming, R, M, C, L );
% get 1st and 2nd derivatives
deriv_1 = [zeros(size(mfcc_tmp,1),1), diff(mfcc_tmp,1,2)];
deriv_2 = [zeros(size(mfcc_tmp,1),2), diff(mfcc_tmp,2,2)];
mfcc_info = [mfcc_tmp;deriv_1;deriv_2]; 

% sync pts and audio.
pts_num = size(m_pts,2);
mfcc_num = size(mfcc_info,2);
two_diff = (mfcc_num / pts_num);
cng = 1;
% mfcc sync.
for in = 1:pts_num
    mfcc_mem(:,in) = mfcc_info(:,floor(cng));
    cng = cng + two_diff;
end


% save it to data structure.
data.all_mfcc = mfcc_info;
data.mfcc = mfcc_mem;
fprintf('DONE: extracting mfcc.\n')

fprintf('DONE: all processes.\n')
end