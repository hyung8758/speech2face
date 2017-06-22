///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010, Jason Mora Saragih, all rights reserved.
//
// This file is part of FaceTracker.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions (licence) and the following disclaimer 
//       in the documentation and/or other materials provided with the 
//       distribution.
//     * The name of the author may not be used to endorse or promote products 
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to 
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite the following work:
//
//       J. M. Saragih, S. Lucey, and J. F. Cohn. Face Alignment through 
//       Subspace Constrained Mean-Shifts. International Conference of Computer 
//       Vision (ICCV), September, 2009.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////
// Eidted by Hyungwon Yang.
// Three information will be retreived from this code. 
// 1. Images, audio, and pts parametes. 
#include <FaceTracker/Tracker.h>
#include <opencv/highgui.h>
#include <OpenAL/al.h>
#include <OpenAL/alc.h>
#include <iostream>
#include <pthread.h>

// How much to capture at a time (affects latency) 
#define CAP_SIZE 2048
#define NUM_THREADS 1
int LOOP_CHECK=1;

const char *save_dir = "/Users/hyungwonyang/Google_Drive/C++/FaceTracker/extracted_data/";

//=============================================================================
void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi)
{
  int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;

  //draw triangulation
  c = CV_RGB(0,0,0);
  for(i = 0; i < tri.rows; i++){
    if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
       visi.at<int>(tri.at<int>(i,1),0) == 0 ||
       visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
		   shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
		   shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
		   shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
		   shape.at<double>(tri.at<int>(i,2)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
		   shape.at<double>(tri.at<int>(i,2)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
		   shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
  }
  //draw connections
  c = CV_RGB(0,0,255);
  for(i = 0; i < con.cols; i++){
    if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
       visi.at<int>(con.at<int>(1,i),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
		   shape.at<double>(con.at<int>(0,i)+n,0));
    p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
		   shape.at<double>(con.at<int>(1,i)+n,0));
    cv::line(image,p1,p2,c,1);
  }
  //draw points
  for(i = 0; i < n; i++){    
    if(visi.at<int>(i,0) == 0)continue;
    p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
    c = CV_RGB(255,0,0); cv::circle(image,p1,2,c);
  }return;
}

//=============================================================================
int parse_cmd(int argc, const char** argv,
	      char* ftFile,char* conFile,char* triFile,
	      bool &fcheck,double &scale,int &fpd)
{
  int i; fcheck = false; scale = 1; fpd = -1;
  for(i = 1; i < argc; i++){
    if((std::strcmp(argv[i],"-?") == 0) ||
       (std::strcmp(argv[i],"--help") == 0)){
      std::cout << "track_face:- Written by Jason Saragih 2010" << std::endl
	   << "Performs automatic face tracking" << std::endl << std::endl
	   << "#" << std::endl 
	   << "# usage: ./face_tracker [options]" << std::endl
	   << "#" << std::endl << std::endl
	   << "Arguments:" << std::endl
	   << "-m <string> -> Tracker model (default: ../model/face2.tracker)"
	   << std::endl
	   << "-c <string> -> Connectivity (default: ../model/face.con)"
	   << std::endl
	   << "-t <string> -> Triangulation (default: ../model/face.tri)"
	   << std::endl
	   << "-s <double> -> Image scaling (default: 1)" << std::endl
	   << "-d <int>    -> Frames/detections (default: -1)" << std::endl
	   << "--check     -> Check for failure" << std::endl;
      return -1;
    }
  }
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"--check") == 0){fcheck = true; break;}
  }
  if(i >= argc)fcheck = false;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-s") == 0){
      if(argc > i+1)scale = std::atof(argv[i+1]); else scale = 1;
      break;
    }
  }
  if(i >= argc)scale = 1;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-d") == 0){
      if(argc > i+1)fpd = std::atoi(argv[i+1]); else fpd = -1;
      break;
    }
  }
  if(i >= argc)fpd = -1;
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-m") == 0){
      if(argc > i+1)std::strcpy(ftFile,argv[i+1]);
      else strcpy(ftFile,"../model/face2.tracker");
      break;
    }
  }
  if(i >= argc)std::strcpy(ftFile,"../model/face2.tracker");
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-c") == 0){
      if(argc > i+1)std::strcpy(conFile,argv[i+1]);
      else strcpy(conFile,"../model/face.con");
      break;
    }
  }
  if(i >= argc)std::strcpy(conFile,"../model/face.con");
  for(i = 1; i < argc; i++){
    if(std::strcmp(argv[i],"-t") == 0){
      if(argc > i+1)std::strcpy(triFile,argv[i+1]);
      else strcpy(triFile,"../model/face.tri");
      break;
    }
  }
  if(i >= argc)std::strcpy(triFile,"../model/face.tri");
  return 0;
}

//=============================================================================
// audio

void *audioProcess(void*)
{
  const int SRATE = 16000;
  ALenum errorCode=0;                                                                            
  // A buffer to hold captured audio
  short buffer[SRATE*2];                                                                                               
  // How many samples are captured 
  ALCint samplesIn=0;                                                                                                

  const char* devices = alcGetString(NULL,ALC_DEFAULT_ALL_DEVICES_SPECIFIER);

  // open wave file for saving.
  std::string audio_name = std::string(save_dir) + "sample.pcm";
  FILE *afile = fopen(audio_name.c_str(),"wb");
  //alGetError();

  // get recording device.
  // Request default audio device 
  ALCdevice* audioDevice = alcOpenDevice(devices);                                                                   
  errorCode = alcGetError(audioDevice); 
  // Create the audio context
  ALCcontext* audioContext = alcCreateContext(audioDevice,NULL);                                                     
  alcMakeContextCurrent(audioContext); 
  errorCode = alcGetError(audioDevice);
  // Request the default capture device with a half-second buffer 
  ALCdevice *device = alcCaptureOpenDevice(NULL, SRATE, AL_FORMAT_MONO16, SRATE/2);
  errorCode = alcGetError(device);
  alcCaptureStart(device);
  errorCode = alcGetError(device);

  while(LOOP_CHECK){
    // Poll for captured audio 
    alcGetIntegerv(device,ALC_CAPTURE_SAMPLES,1,&samplesIn);                                            
    if (samplesIn>CAP_SIZE) { 
      // Grab the sound 
      alcCaptureSamples(device,buffer,samplesIn);                                                          
      //printf("sample numbers: %d\n",samplesIn);
      fwrite(buffer, sizeof(short), samplesIn, afile);
      } 
  }
  fclose(afile);
  // stop capture 
  alcCaptureStop(device);                                                                                       
  alcCaptureCloseDevice(device);
  // clean up
  errorCode = alGetError(); 
  alcMakeContextCurrent(NULL); 
  errorCode = alGetError(); 
  alcDestroyContext(audioContext); 
  alcCloseDevice(audioDevice); 

  pthread_exit(NULL);
}

//=============================================================================
int main(int argc, const char** argv)
{ 
  // file open for saving parameters
  std::string param_name = std::string(save_dir) + "parameters.txt";
  FILE* fp = fopen(param_name.c_str(),"wt");

  //parse command line arguments
  char ftFile[256],conFile[256],triFile[256];
  bool fcheck = false; double scale = 1; int fpd = -1; bool show = true;
  if(parse_cmd(argc,argv,ftFile,conFile,triFile,fcheck,scale,fpd)<0)return 0;

  //set other tracking parameters
  std::vector<int> wSize1(1); wSize1[0] = 7;
  std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
  int nIter = 5; double clamp=3,fTol=0.01; 
  FACETRACKER::Tracker model(ftFile);
  cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
  cv::Mat con=FACETRACKER::IO::LoadCon(conFile);
  
  //initialize camera and display window
  cv::Mat frame,gray,im; double fps=0; char sss[256]; std::string text; 
  cv::VideoCapture camera(CV_CAP_ANY); if(!camera.isOpened()) return -1;
  int64 t1,t0 = cvGetTickCount(); int fnum=0;
  cvNamedWindow("Face Tracker",1);
  std::cout << "Hot keys: "        << std::endl
	    << "\t ESC - quit"     << std::endl
	    << "\t d   - Redetect" << std::endl;

  //loop until quit (i.e user presses ESC)
  bool failed = true;
  int reco=0;

  // audio recordning thread.
  //pthread_attr_t  attr;
  pthread_t threads;
  int pth; 
  std::cout << "audio recording is activated.\n";
  pth = pthread_create(&threads, NULL, &audioProcess, NULL);
  if (pth){
         std::cout << "Error:unable to create thread," << pth << std::endl;
         exit(-1);
      }

  while(1){ 
    //grab image, resize and flip
    reco +=1;
    camera.read(frame);
    if(scale == 1)im = frame; 
    else cv::resize(frame,im,cv::Size(scale*frame.cols,scale*frame.rows));
    cv::flip(im,im,1); cv::cvtColor(im,gray,CV_BGR2GRAY);
    
    // save image file.
    // char buf[4];
    // sprintf(buf, "%04d",reco);
    // std::string image_name = save_dir + "image/img_" + std::string(buf) + ".png";
    // write image file
    // cv::imwrite(img_name,im);

    //track this image
    std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1; 
    if(model.Track(gray,wSize,fpd,nIter,clamp,fTol,fcheck) == 0){
      int idx = model._clm.GetViewIdx(); failed = false;
  

    int n = model._shape.rows/2;
      for(int i = 0; i < n; i++){
        if(model._clm._visi[idx].at<int>(i,0) == 0) continue;
          cv::Point p1 = cv::Point(model._shape.at<double>(i,0),model._shape.at<double>(i+n,0));
          // c = CV_RGB(255,0,0); cv::circle(image,p1,2,c);
          fprintf(fp,"%d,%d \n",p1.x,p1.y);
          }

      Draw(im,model._shape,con,tri,model._clm._visi[idx]); 
    }else{
      if(show){cv::Mat R(im,cvRect(0,0,150,50)); R = cv::Scalar(0,0,255);}
      model.FrameReset(); failed = true;
    }     
    //draw framerate on display image 
    if(fnum >= 9){      
      t1 = cvGetTickCount();
      fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6); 
      t0 = t1; fnum = 0;
    }else fnum += 1;
    if(show){
      sprintf(sss,"%d frames/sec",(int)round(fps)); text = sss;
      cv::putText(im,text,cv::Point(10,20),
		  CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
    }
    //show image and check for user input
    imshow("Face Tracker",im); 
    int c = cvWaitKey(10);
    if(c == 27)break; else if(char(c) == 'd')model.FrameReset();
    fprintf(fp,"-------parameter end------\n");
  } 
  LOOP_CHECK = 0;

  // close text and audio file.
  fclose(fp);
  pthread_exit(NULL);
  return 0;
}
//=============================================================================
