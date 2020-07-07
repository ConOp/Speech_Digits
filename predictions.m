
%{
[y, fs] = audioread(sprintf('digits_audio.mp3')); 
fs = 8000 ; 
nBits = 16 ; 
nChannels = 1 ; 
ID = -1; % default audio input device 
rec = audiorecorder(fs,nBits,nChannels,ID); 
recordblocking(rec,5);
recording = getaudiodata(rec);
audiowrite('test.wav',recording,fs);
%}

function digits=predictions(filename)
    model = loadLearnerForCoder('BestAttempt');
    [y,fs]=audioread(filename);
    envelope = imdilate(abs(y), true(1500, 1));
    quietParts = envelope > 0.06;
    beginning = strfind(quietParts',[0 1]);
    ending = strfind(quietParts', [1 0]);

    parts = cell(numel(beginning),1);
    for i=1:numel(beginning)
        parts{i}=y(beginning(i):ending(i));
    end
    results = zeros(numel(parts),1);
    for i=1:numel(parts)
        aFE = audioFeatureExtractor(...
        "SampleRate",fs, ...
        "Window",hamming(round(0.1*fs),"periodic"), ...
        "OverlapLength",round(0.02*fs), ...
        "mfcc",true, ...
        "mfccDelta",true, ...
        "mfccDeltaDelta",true, ...
        "pitch",true, ...
        "spectralCentroid",true);

        feature_array = extract(aFE,parts{i});

        [~, score] = predict(model,feature_array);
        [m,i1] = max(score,[],2);

        [~,i2] = max(m);
        results(i) = i1(i2)-1;
        
    end
    digits=results;
end

