function playRawAudio(data)

wave = data.wave;
sr = data.wave_srate;

% play sound
soundsc(wave,sr)

end