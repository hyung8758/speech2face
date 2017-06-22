% Running tutorial: tutorial.m
% Please follow the steps below
%
%                                                           06.21.17
%                                                      Hyungwon Yang

%%% prerequisite
% Before running this script, Extract image and audio data 
% through FaceTracker or OpenFace.
% This tutorial assumed that the preprocessable raw data is stored in the 
% ./data directory.

%% Path setting.
addpath(genpath('data'))
addpath(genpath('extractors'))
addpath('src')

%% 1st step: play with the sample data.
% In this step, we will play with the sample data first. 
% Notice that the raw data(image and audio) was recorded from FaceTracker
% toolkit and preprocessed by 'data4train.m' and 'trainData.m'.

% Load the sampel data.
sample_data = load('sample_data');
data = sample_data.sample_data;
%% Check the face indices.
% 1st argv: data structure (it can be obtained by running data4train.m)
% 2nd argv: indexing speed.
showFaceIndex(data,0.1)

%% Check the raw data.
% faceAnimation needs face_data variable on your workspace.
face_data = data;
faceAnimation()

% terminate during plotting
% clear player; close

%% 2nd step: Preprocess raw data.
clear; clc; close all
% If you already extracted raw data from FaceTracker(./extractors/FaceTracker)
% you can use that data in this step.

% Set the data path. You can write your own data path. Remember that a
% direcotry should contain 3 types of data: image(direcotry),
% parameters.txt(text file), and sample.pcm (pcm file) 
% DO NOT change the name of 3 types of data but you can change the name
% of the directory that holds those 3 types of data.
data_path = './data/FaceTrackerData';
data = data4train(data_path);

% structure data information.
%% image : For overload problem, I commented out the saving image lines in
% FaceTracker(line 285~289 in face_tracker.cc). But you can save images by
% uncomment those lines.(In case, you have to recompile it!)
% Since I didn't save images, image field is empty.
images = data.image;

% param: Face pts parameters.
param = data.param;

% wave: Raw audio signal.
wave = data.wave;
% playRawAudio(data) % play sound
% clear sound % stop sound.

% wave_srate: Audio sampling rate.
wave_srate = data.wave_srate;

% face_pts: 66(total 132 x and y points) face pts points.
face_pts =  data.face_pts;

% mouth_pts: 18(total 36 x and y points) mouth pts points.
mouth_pts = data.mouth_pts;

% all_mfcc: mfcc values from the raw audio file.
all_mfcc =  data.all_mfcc;

% all_mfcc: extracted mfcc values in order to sync with the param data.
extracted_mfcc =  data.mfcc;

%% 3rd step: training dataset.
% edit trainData function if you would like to change number of hidden
% layers and units or other parameters.
% edit trainData 
% net = trainData(extracted_mfcc,face_pts);

%% 4th step: Initiatie speech2lips.
% First you need to open speech2lips and set the trained network path.
% Default path is already set in the script but you can change it whenever
% your trained network is ready.
% edit speech2lips
speech2lips



