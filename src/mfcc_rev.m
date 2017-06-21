function [ CC, FBE, frames , center_time ] = mfcc_rev( speech, fs, Tw, Ts, alpha, window, R, M, N, L )
% MFCC Mel frequency cepstral coefficient feature extraction.
%
%   edited by JK (2015-8-23)
%
%   MFCC(S,FS,TW,TS,ALPHA,WINDOW,R,M,N,L) returns mel frequency 
%   cepstral coefficients (MFCCs) computed from speech signal given 
%   in vector S and sampled at FS (Hz). The speech signal is first 
%   preemphasised using a first order FIR filter with preemphasis 
%   coefficient ALPHA. The preemphasised speech signal is subjected 
%   to the short-time Fourier transform analysis with frame durations 
%   of TW (ms), frame shifts of TS (ms) and analysis window function 
%   given as a function handle in WINDOW. This is followed by magnitude 
%   spectrum computation followed by filterbank design with M triangular 
%   filters uniformly spaced on the mel scale between lower and upper 
%   frequency limits given in R (Hz). The filterbank is applied to 
%   the magnitude spectrum values to produce filterbank energies (FBEs) 
%   (M per frame). Log-compressed FBEs are then decorrelated using the 
%   discrete cosine transform to produce cepstral coefficients. Final
%   step applies sinusoidal lifter to produce liftered MFCCs that 
%   closely match those produced by HTK [1].
%
%   [CC,FBE,FRAMES]=MFCC(...) also returns FBEs and windowed frames,
%   with feature vectors and frames as columns.
%
%   This framework is based on Dan Ellis' rastamat routines [2]. The 
%   emphasis is placed on closely matching MFCCs produced by HTK [1]
%   (refer to p.337 of [1] for HTK's defaults) with simplicity and 
%   compactness as main considerations, but at a cost of reduced 
%   flexibility. This routine is meant to be easy to extend, and as 
%   a starting point for work with cepstral coefficients in MATLAB.
%   The triangular filterbank equations are given in [3].
%
%   Inputs
%           S is the input speech signal (as vector)
%
%           FS is the sampling frequency (Hz) 
%
%           TW is the analysis frame duration (ms) 
% 
%           TS is the analysis frame shift (ms)
%
%           ALPHA is the preemphasis coefficient
%
%           WINDOW is a analysis window function handle
% 
%           R is the frequency range (Hz) for filterbank analysis
%
%           M is the number of filterbank channels
%
%           N is the number of cepstral coefficients 
%             (including the 0th coefficient)
%
%           L is the liftering parameter
%
%   Outputs
%           CC is a matrix of mel frequency cepstral coefficients
%              (MFCCs) with feature vectors as columns
%
%           FBE is a matrix of filterbank energies
%               with feature vectors as columns
%
%           FRAMES is a matrix of windowed frames
%                  (one frame per column)
%
%   Example
%           Tw = 25;           % analysis frame duration (ms)
%           Ts = 10;           % analysis frame shift (ms)
%           alpha = 0.97;      % preemphasis coefficient
%           R = [ 300 3700 ];  % frequency range to consider
%           M = 20;            % number of filterbank channels 
%           C = 13;            % number of cepstral coefficients
%           L = 22;            % cepstral sine lifter parameter
%       
%           % hamming window (see Eq. (5.2) on p.73 of [1])
%           hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));
%       
%           % Read speech samples, sampling rate and precision from file
%           [ speech, fs, nbits ] = wavread( 'sp10.wav' );
%       
%           % Feature extraction (feature vectors as columns)
%           [ MFCCs, FBEs, frames ] = ...
%                           mfcc( speech, fs, Tw, Ts, alpha, hamming, R, M, C, L );
%       
%           % Plot cepstrum over time
%           figure('Position', [30 100 800 200], 'PaperPositionMode', 'auto', ... 
%                  'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
%       
%           imagesc( [1:size(MFCCs,2)], [0:C-1], MFCCs ); 
%           axis( 'xy' );
%           xlabel( 'Frame index' ); 
%           ylabel( 'Cepstrum index' );
%           title( 'Mel frequency cepstrum' );
%
%   References
%
%           [1] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., 
%               Liu, X., Moore, G., Odell, J., Ollason, D., Povey, D., 
%               Valtchev, V., Woodland, P., 2006. The HTK Book (for HTK 
%               Version 3.4.1). Engineering Department, Cambridge University.
%               (see also: http://htk.eng.cam.ac.uk)
%
%           [2] Ellis, D., 2005. Reproducing the feature outputs of 
%               common programs using Matlab and melfcc.m. url: 
%               http://labrosa.ee.columbia.edu/matlab/rastamat/mfccs.html
%
%           [3] Huang, X., Acero, A., Hon, H., 2001. Spoken Language 
%               Processing: A guide to theory, algorithm, and system 
%               development. Prentice Hall, Upper Saddle River, NJ, 
%               USA (pp. 314-315).
%
%   See also EXAMPLE, COMPARE, FRAMES2VEC, TRIFBANK.

%   Author: Kamil Wojcicki, September 2011


    %% PRELIMINARIES 

    % Ensure correct number of inputs
    if( nargin~= 10 ), help mfcc; return; end; 

    % Explode samples to the range of 16 bit shorts
    if( max(abs(speech))<=1 ), speech = speech * 2^15; end;

    Nw = round( 1E-3*Tw*fs );    % frame duration (samples)
    Ns = round( 1E-3*Ts*fs );    % frame shift (samples)

    nfft = 2^nextpow2( Nw );     % length of FFT analysis 
    K = nfft/2+1;                % length of the unique part of the FFT 


    %% HANDY INLINE FUNCTION HANDLES

    % Forward and backward mel frequency warping (see Eq. (5.13) on p.76 of [1]) 
    % Note that base 10 is used in [1], while base e is used here and in HTK code
    hz2mel = @( hz )( 1127*log(1+hz/700) );     % Hertz to mel warping function
    mel2hz = @( mel )( 700*exp(mel/1127)-700 ); % mel to Hertz warping function

    % Type III DCT matrix routine (see Eq. (5.14) on p.77 of [1])
    dctm = @( N, M )( sqrt(2.0/M) * cos( repmat([0:N-1].',1,M) ...
                                       .* repmat(pi*([1:M]-0.5)/M,N,1) ) );

    % Cepstral lifter routine (see Eq. (5.12) on p.75 of [1])
    ceplifter = @( N, L )( 1+0.5*L*sin(pi*[0:N-1]/L) );


    %% FEATURE EXTRACTION 

    % Preemphasis filtering (see Eq. (5.1) on p.73 of [1])
    speech = filter( [1 -alpha], 1, speech ); % fvtool( [1 -alpha], 1 );

    % Framing and windowing (frames as columns)
    [frames,indexes] = vec2frames( speech, Nw, Ns, 'cols', window, false );
    center_time = indexes(round(size(indexes,1)/2),:)/fs*1000;
    
    % Magnitude spectrum computation (as column vectors)
    MAG = abs( fft(frames,nfft,1) ); 

    % Triangular filterbank with uniformly spaced filters on mel scale
    H = trifbank( M, K, R, fs, hz2mel, mel2hz ); % size of H is M x K 

    % Filterbank application to unique part of the magnitude spectrum
    FBE = H * MAG(1:K,:); % FBE( FBE<1.0 ) = 1.0; % apply mel floor

    % DCT matrix computation
    DCT = dctm( N, M );

    % Conversion of logFBEs to cepstral coefficients through DCT
    CC =  DCT * log( FBE );

    % Cepstral lifter computation
    lifter = ceplifter( N, L );

    % Cepstral liftering gives liftered cepstral coefficients
    CC = diag( lifter ) * CC; % ~ HTK's MFCCs

% EOF


function [ H, f, c ] = trifbank( M, K, R, fs, h2w, w2h )
% TRIFBANK Triangular filterbank.
%
%   [H,F,C]=TRIFBANK(M,K,R,FS,H2W,W2H) returns matrix of M triangular filters 
%   (one per row), each K coefficients long along with a K coefficient long 
%   frequency vector F and M+2 coefficient long cutoff frequency vector C. 
%   The triangular filters are between limits given in R (Hz) and are 
%   uniformly spaced on a warped scale defined by forward (H2W) and backward 
%   (W2H) warping functions.
%
%   Inputs
%           M is the number of filters, i.e., number of rows of H
%
%           K is the length of frequency response of each filter 
%             i.e., number of columns of H
%
%           R is a two element vector that specifies frequency limits (Hz), 
%             i.e., R = [ low_frequency high_frequency ];
%
%           FS is the sampling frequency (Hz)
%
%           H2W is a Hertz scale to warped scale function handle
%
%           W2H is a wared scale to Hertz scale function handle
%
%   Outputs
%           H is a M by K triangular filterbank matrix (one filter per row)
%
%           F is a frequency vector (Hz) of 1xK dimension
%
%           C is a vector of filter cutoff frequencies (Hz), 
%             note that C(2:end) also represents filter center frequencies,
%             and the dimension of C is 1x(M+2)
%
%   Example
%           fs = 16000;               % sampling frequency (Hz)
%           nfft = 2^12;              % fft size (number of frequency bins)
%           K = nfft/2+1;             % length of each filter
%           M = 23;                   % number of filters
%
%           hz2mel = @(hz)(1127*log(1+hz/700)); % Hertz to mel warping function
%           mel2hz = @(mel)(700*exp(mel/1127)-700); % mel to Hertz warping function
%
%           % Design mel filterbank of M filters each K coefficients long,
%           % filters are uniformly spaced on the mel scale between 0 and Fs/2 Hz
%           [ H1, freq ] = trifbank( M, K, [0 fs/2], fs, hz2mel, mel2hz );
%
%           % Design mel filterbank of M filters each K coefficients long,
%           % filters are uniformly spaced on the mel scale between 300 and 3750 Hz
%           [ H2, freq ] = trifbank( M, K, [300 3750], fs, hz2mel, mel2hz );
%
%           % Design mel filterbank of 18 filters each K coefficients long, 
%           % filters are uniformly spaced on the Hertz scale between 4 and 6 kHz
%           [ H3, freq ] = trifbank( 18, K, [4 6]*1E3, fs, @(h)(h), @(h)(h) );
%
%            hfig = figure('Position', [25 100 800 600], 'PaperPositionMode', ...
%                              'auto', 'Visible', 'on', 'color', 'w'); hold on; 
%           subplot( 3,1,1 ); 
%           plot( freq, H1 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%       
%           subplot( 3,1,2 );
%           plot( freq, H2 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%       
%           subplot( 3,1,3 ); 
%           plot( freq, H3 );
%           xlabel( 'Frequency (Hz)' ); ylabel( 'Weight' ); set( gca, 'box', 'off' ); 
%
%   Reference
%           [1] Huang, X., Acero, A., Hon, H., 2001. Spoken Language Processing: 
%               A guide to theory, algorithm, and system development. 
%               Prentice Hall, Upper Saddle River, NJ, USA (pp. 314-315).

%   Author  Kamil Wojcicki, UTD, June 2011


    if( nargin~= 6 ), help trifbank; return; end; % very lite input validation

    f_min = 0;          % filter coefficients start at this frequency (Hz)
    f_low = R(1);       % lower cutoff frequency (Hz) for the filterbank 
    f_high = R(2);      % upper cutoff frequency (Hz) for the filterbank 
    f_max = 0.5*fs;     % filter coefficients end at this frequency (Hz)
    f = linspace( f_min, f_max, K ); % frequency range (Hz), size 1xK
    fw = h2w( f );

    % filter cutoff frequencies (Hz) for all filters, size 1x(M+2)
    c = w2h( h2w(f_low)+[0:M+1]*((h2w(f_high)-h2w(f_low))/(M+1)) );
    cw = h2w( c );

    H = zeros( M, K );                  % zero otherwise
    for m = 1:M 

        % implements Eq. (6.140) on page 314 of [1] 
        % k = f>=c(m)&f<=c(m+1); % up-slope
        % H(m,k) = 2*(f(k)-c(m)) / ((c(m+2)-c(m))*(c(m+1)-c(m)));
        % k = f>=c(m+1)&f<=c(m+2); % down-slope
        % H(m,k) = 2*(c(m+2)-f(k)) / ((c(m+2)-c(m))*(c(m+2)-c(m+1)));

        % implements Eq. (6.141) on page 315 of [1]
        k = f>=c(m)&f<=c(m+1); % up-slope
        H(m,k) = (f(k)-c(m))/(c(m+1)-c(m));
        k = f>=c(m+1)&f<=c(m+2); % down-slope
        H(m,k) = (c(m+2)-f(k))/(c(m+2)-c(m+1));
       
   end

   % H = H./repmat(max(H,[],2),1,K);  % normalize to unit height (inherently done)
   % H = H./repmat(trapz(f,H,2),1,K); % normalize to unit area 


% EOF

function [ frames, indexes ] = vec2frames( vec, Nw, Ns, direction, window, padding )
% VEC2FRAMES Splits signal into overlapped frames using indexing.
% 
%   B=vec2frames(A,M,N) creates a matrix B whose columns consist of 
%   segments of length M, taken at every N samples along input vector A.
%
%   [B,R]=vec2frames(A,M,N,D,W,P) creates a matrix B whose columns 
%   or rows, as specified by D, consist of segments of length M, taken 
%   at every N samples along the input vector A and windowed using the
%   analysis window specified by W. The division of A into frames is 
%   achieved using indexes returned in R as follows: B=A(R);
%
%   Summary
%
%           A is an input vector
%
%           M is a frame length (in samples)
%
%           N is a frame shift (in samples)
%
%           D specifies if the frames in B are rows or columns,
%             i.e., D = 'rows' or 'cols', respectively
%
%           W is an optional analysis window function to be applied to 
%             each frame, given as a function handle, e.g., W = @hanning
%             or as a vector of window samples, e.g., W = hanning( M )
%
%           P specifies if last frame should be padded to full length,
%             or simply discarded, i.e., P = true or false, respectively
%   
%           B is the output matrix of frames
%
%           R is a matrix of indexes used for framing, such that division 
%             of A into frames is achieved as follows: B=A(R);
%
%   Examples
%
%           % divide the input vector into seven-sample-long frames with a shift
%           % of three samples and return frames as columns of the output matrix
%           % (note that the last sample of the input vector is discarded)
%           vec2frames( [1:20], 7, 3 )
%
%           % divide the input vector into seven-sample-long frames with a shift
%           % of three samples and return frames as rows of the output matrix
%           % (note that the last sample of the input vector is discarded)
%           vec2frames( [1:20], 7, 3, 'rows' )
%
%           % divide the input vector into seven-sample-long frames with a shift
%           % of three samples, pad the last frame with zeros so that no samples
%           % are discarded and return frames as rows of the output matrix
%           vec2frames( [1:20], 7, 3, 'rows', [], true )
%
%           % divide the input vector into seven-sample-long frames with a shift
%           % of three samples, pad the last frame with white Gaussian noise
%           % of variance (1E-5)^2 so that no samples are discarded and 
%           % return frames as rows of the output matrix
%           vec2frames( [1:20], 7, 3, 'rows', false, { 'noise', 1E-5 } )
%
%           % divide the input vector into seven-sample-long frames with a shift
%           % of three samples, pad the last frame with zeros so that no samples 
%           % are discarded, apply the Hanning analysis window to each frame and
%           % return frames as columns of the output matrix
%           vec2frames( [1:20], 7, 3, 'cols', @hanning, 0 )
% 
%   See also FRAMES2VEC, DEMO

%   Author: Kamil Wojcicki, UTD, July 2011


    % usage information
    usage = 'usage: [ frames, indexes ] = vec2frames( vector, frame_length, frame_shift, direction, window, padding );';

    % default settings 
    switch( nargin )
    case { 0, 1, 2 }, error( usage );
    case 3, padding=false; window=false; direction='cols';
    case 4, padding=false; window=false; 
    case 5, padding=false; 
    end

    % input validation
    if( isempty(vec) || isempty(Nw) || isempty(Ns) ), error( usage ); end;
    if( min(size(vec))~=1 ), error( usage ); end;
    if( Nw==0 || Ns==0 ), error( usage ); end;

    vec = vec(:);                       % ensure column vector

    L = length( vec );                  % length of the input vector
    M = floor((L-Nw)/Ns+1);             % number of frames 


    % perform signal padding to enable exact division of signal samples into frames 
    % (note that if padding is disabled, some samples may be discarded)
    if( ~isempty(padding) )
 
        % figure out if the input vector can be divided into frames exactly
        E = (L-((M-1)*Ns+Nw));

        % see if padding is actually needed
        if( E>0 ) 

            % how much padding will be needed to complete the last frame?
            P = Nw-E;

            % pad with zeros
            if( islogical(padding) && padding ) 
                vec = [ vec; zeros(P,1) ];

            % pad with a specific numeric constant
            elseif( isnumeric(padding) && length(padding)==1 ) 
                vec = [ vec; padding*ones(P,1) ];

            % pad with a low variance white Gaussian noise
            elseif( isstr(padding) && strcmp(padding,'noise') ) 
                vec = [ vec; 1E-6*randn(P,1) ];

            % pad with a specific variance white Gaussian noise
            elseif( iscell(padding) && strcmp(padding{1},'noise') ) 
                if( length(padding)>1 ), scale = padding{2}; 
                else, scale = 1E-6; end;
                vec = [ vec; scale*randn(P,1) ];

            % if not padding required, decrement frame count
            % (not a very elegant solution)
            else
                M = M-1;

            end

            % increment the frame count
            M = M+1;
        end
    end


    % compute index matrix 
    switch( direction )

    case 'rows'                                                 % for frames as rows
        indf = Ns*[ 0:(M-1) ].';                                % indexes for frames      
        inds = [ 1:Nw ];                                        % indexes for samples
        indexes = indf(:,ones(1,Nw)) + inds(ones(M,1),:);       % combined framing indexes
    
    case 'cols'                                                 % for frames as columns
        indf = Ns*[ 0:(M-1) ];                                  % indexes for frames      
        inds = [ 1:Nw ].';                                      % indexes for samples
        indexes = indf(ones(Nw,1),:) + inds(:,ones(1,M));       % combined framing indexes
    
    otherwise
        error( sprintf('Direction: %s not supported!\n', direction) ); 

    end


    % divide the input signal into frames using indexing
    frames = vec( indexes );


    % return if custom analysis windowing was not requested
    if( isempty(window) || ( islogical(window) && ~window ) ), return; end;
    
    % if analysis window function handle was specified, generate window samples
    if( isa(window,'function_handle') )
        window = window( Nw );
    end
    
    % make sure analysis window is numeric and of correct length, otherwise return
    if( isnumeric(window) && length(window)==Nw )

        % apply analysis windowing beyond the implicit rectangular window function
        switch( direction )
        case 'rows', frames = frames * diag( window );
        case 'cols', frames = diag( window ) * frames;
        end

    end


% EOF 
